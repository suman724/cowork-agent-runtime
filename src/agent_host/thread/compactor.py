"""Context compaction — drop oldest messages to fit token budget."""

from __future__ import annotations

import abc
from typing import Any

from agent_host.thread.token_counter import estimate_message_tokens


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
