"""Context compaction — drop oldest messages to fit token budget."""

from __future__ import annotations

import abc
import json
from typing import TYPE_CHECKING, Any

import structlog

from agent_host.thread.token_counter import estimate_message_tokens

if TYPE_CHECKING:
    from agent_host.llm.client import LLMClient

logger = structlog.get_logger()


class ContextCompactor(abc.ABC):
    """Abstract base for context compaction strategies."""

    @abc.abstractmethod
    def compact(self, messages: list[dict[str, Any]], budget_tokens: int) -> list[dict[str, Any]]:
        """Compact messages to fit within token budget.

        Args:
            messages: Full message list (system prompt first).
            budget_tokens: Maximum token estimate to target.

        Returns:
            Compacted message list.
        """


class DropOldestCompactor(ContextCompactor):
    """Keep system prompt + recent N messages, drop oldest from middle.

    Same algorithm as the prior context_truncation.py but for OpenAI message format.
    Inserts a marker message when messages are dropped.
    """

    def __init__(self, recency_window: int = 20) -> None:
        self._recency_window = recency_window

    def compact(self, messages: list[dict[str, Any]], budget_tokens: int) -> list[dict[str, Any]]:
        """Drop oldest middle messages to fit within token budget."""
        if not messages:
            return messages

        total = sum(estimate_message_tokens(m) for m in messages)
        if total <= budget_tokens:
            return messages

        # Keep first message (system prompt) + last recency_window messages
        keep_head = messages[:1]
        if len(messages) <= self._recency_window + 1:
            keep_tail = messages[1:]
            middle: list[dict[str, Any]] = []
        else:
            keep_tail = messages[-self._recency_window :]
            middle = list(messages[1 : -self._recency_window])

        dropped = 0
        while middle and self._estimate_total(keep_head, middle, keep_tail) > budget_tokens:
            middle.pop(0)
            dropped += 1

        # Insert marker if messages were dropped
        if dropped > 0:
            marker: dict[str, Any] = {
                "role": "system",
                "content": f"[... {dropped} earlier messages omitted ...]",
            }
            return [*keep_head, marker, *middle, *keep_tail]

        return [*keep_head, *middle, *keep_tail]

    @staticmethod
    def _estimate_total(
        head: list[dict[str, Any]],
        middle: list[dict[str, Any]],
        tail: list[dict[str, Any]],
    ) -> int:
        """Estimate total tokens across three segments."""
        total = 0
        for msg in head:
            total += estimate_message_tokens(msg)
        for msg in middle:
            total += estimate_message_tokens(msg)
        for msg in tail:
            total += estimate_message_tokens(msg)
        return total


_SUMMARIZATION_PROMPT = (
    "Summarize the following conversation segment. This summary will replace "
    "the original messages in the context window, so it must preserve all "
    "information needed to continue the task correctly.\n\n"
    "Preserve:\n"
    "- File paths and directories examined or modified\n"
    "- Key decisions made and their reasoning\n"
    "- Errors encountered and resolutions\n"
    "- Data values, calculations, or results that may be referenced later\n"
    "- Current approach and strategy\n\n"
    "Format as a concise structured summary (200-400 words max).\n\n"
    "Conversation segment to summarize:\n"
)


class HybridCompactor(ContextCompactor):
    """Two-phase compaction: observation masking + optional LLM summarization.

    Phase 1 (observation masking): Replace old tool result messages with one-line
    summaries. Preserves assistant tool_call messages. Typically achieves 50-70%
    compression since tool outputs are the largest messages.

    Phase 2 (LLM summarization): When masking alone can't fit the budget, summarize
    the oldest masked messages into a single system message. Requires async
    pre-computation via ``precompute_summary()`` before ``compact()`` is called.
    """

    def __init__(
        self,
        recency_window: int = 20,
        mask_only: bool = False,
    ) -> None:
        self._recency_window = recency_window
        self._mask_only = mask_only
        self._cached_summary: str | None = None
        self._summary_covers_up_to: int = 0

    def compact(self, messages: list[dict[str, Any]], budget_tokens: int) -> list[dict[str, Any]]:
        """Synchronous compaction: masking + cached summary."""
        if not messages:
            return messages

        # Phase 1: observation masking
        masked = self._mask_old_observations(messages)
        total = sum(estimate_message_tokens(m) for m in masked)
        if total <= budget_tokens:
            return masked

        # Phase 2: apply cached summary if available
        if self._cached_summary:
            result = self._apply_cached_summary(masked, budget_tokens)
            if result is not None:
                return result

        # Fallback: drop-oldest (same as DropOldestCompactor)
        return DropOldestCompactor(self._recency_window).compact(masked, budget_tokens)

    async def precompute_summary(
        self,
        messages: list[dict[str, Any]],
        budget_tokens: int,
        llm_client: LLMClient,
    ) -> None:
        """Async pre-computation: generate LLM summary if masking isn't enough."""
        if self._mask_only:
            return

        masked = self._mask_old_observations(messages)
        total = sum(estimate_message_tokens(m) for m in masked)
        if total <= budget_tokens:
            return  # masking is sufficient

        # Identify messages to summarize (oldest beyond recency window)
        to_summarize = self._select_for_summary(masked, budget_tokens)
        if not to_summarize:
            return

        try:
            self._cached_summary = await self._generate_summary(to_summarize, llm_client)
            self._summary_covers_up_to = len(messages) - self._recency_window
            logger.info(
                "compaction_summary_generated",
                messages_summarized=len(to_summarize),
            )
        except Exception:
            logger.warning("compaction_summary_failed", exc_info=True)
            self._cached_summary = None

    def _mask_old_observations(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace old tool results with one-line summaries."""
        cutoff = max(1, len(messages) - self._recency_window)
        result: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            if i > 0 and i < cutoff and msg.get("role") == "tool":
                tool_name = msg.get("name", "tool")
                content = msg.get("content", "")
                masked_content = self._mask_tool_output(tool_name, content)
                result.append({**msg, "content": masked_content})
            else:
                result.append(msg)
        return result

    @staticmethod
    def _mask_tool_output(tool_name: str, content: str) -> str:
        """Generate a one-line summary of a tool output without LLM."""
        try:
            data = json.loads(content)
            status = data.get("status", "success")
            output = data.get("output", "")
        except (json.JSONDecodeError, TypeError):
            output = str(content)
            status = "success"

        if status == "denied":
            return f"[{tool_name}: denied]"
        if status == "failed":
            err_msg = ""
            if isinstance(data, dict):
                err_msg = str(data.get("error", {}).get("message", ""))[:80]
            return f"[{tool_name}: failed — {err_msg}]"

        lines = output.count("\n") + 1 if output else 0
        chars = len(output)
        return f"[{tool_name}: {status}, {lines} lines / {chars} chars]"

    def _select_for_summary(
        self, masked: list[dict[str, Any]], _budget_tokens: int
    ) -> list[dict[str, Any]]:
        """Select the oldest messages beyond recency window for summarization."""
        if len(masked) <= self._recency_window + 1:
            return []
        # Everything between system prompt and recency window
        return list(masked[1 : -self._recency_window])

    @staticmethod
    async def _generate_summary(messages: list[dict[str, Any]], llm_client: LLMClient) -> str:
        """Call LLM to summarize a conversation segment."""
        # Format messages for the summarization prompt
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))[:500]
            parts.append(f"[{role}]: {content}")
        segment_text = "\n".join(parts)

        summary_messages = [
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": _SUMMARIZATION_PROMPT + segment_text},
        ]

        response = await llm_client.stream_chat(summary_messages, tools=None)
        return response.text

    def _apply_cached_summary(
        self, masked: list[dict[str, Any]], budget_tokens: int
    ) -> list[dict[str, Any]] | None:
        """Replace oldest messages with cached summary."""
        if not self._cached_summary:
            return None

        head = masked[:1]  # system prompt
        tail = masked[-self._recency_window :] if len(masked) > self._recency_window else masked[1:]
        summary_msg: dict[str, Any] = {
            "role": "system",
            "content": f"[Summary of earlier conversation]\n\n{self._cached_summary}",
        }

        result = [*head, summary_msg, *tail]
        total = sum(estimate_message_tokens(m) for m in result)
        if total <= budget_tokens:
            return result
        return None
