"""LLM Gateway streaming client using the OpenAI SDK."""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any

import structlog
from openai import AsyncOpenAI

from agent_host.exceptions import LLMGatewayError
from agent_host.llm.error_classifier import (
    extract_retry_after,
    is_rate_limit_error,
    is_transient_llm_error,
)
from agent_host.llm.models import LLMResponse, ToolCallMessage
from agent_host.thread.token_counter import estimate_message_tokens, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_host.events.event_emitter import EventEmitter

logger = structlog.get_logger()


class LLMClient:
    """Streaming client for an OpenAI-compatible LLM Gateway.

    Uses ``AsyncOpenAI`` to stream chat completions. Retries transient errors
    with exponential backoff + jitter (reuses ``error_classifier.py``).
    """

    def __init__(
        self,
        endpoint: str,
        auth_token: str,
        model: str,
        *,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        timeout: float = 120.0,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=endpoint,
            api_key=auth_token,
            timeout=timeout,
            max_retries=0,  # We handle retries ourselves
        )
        # Strip LiteLLM-style provider prefix (e.g. "openai/gpt-4o" → "gpt-4o")
        self._model = model.split("/", 1)[-1] if "/" in model else model
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._event_emitter = event_emitter

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        task_id: str = "",
        on_text_chunk: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Stream a chat completion, returning the full parsed response.

        Text deltas are forwarded via the ``on_text_chunk`` callback as they arrive.
        Retries transient errors with exponential backoff + jitter.

        Args:
            messages: OpenAI-format message list.
            tools: OpenAI-format tool definitions (or None to disable tools).
            task_id: Current task ID for event emission.
            on_text_chunk: Called with each text delta as it streams.

        Returns:
            Parsed LLMResponse with text, tool_calls, token usage.
        """
        import asyncio

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._do_stream(messages, tools, on_text_chunk)
            except Exception as exc:
                last_error = exc
                if is_transient_llm_error(exc) and attempt < self._max_retries:
                    # Use Retry-After header for rate limits when available,
                    # otherwise fall back to exponential backoff
                    retry_after = extract_retry_after(exc) if is_rate_limit_error(exc) else None
                    if retry_after is not None:
                        delay = min(retry_after, self._retry_max_delay)
                    elif getattr(exc, "status_code", None) == 529:
                        # 529 Overloaded: longer backoff (3x base)
                        delay = min(
                            self._retry_base_delay * 3 * (2**attempt),
                            self._retry_max_delay,
                        )
                    else:
                        delay = min(
                            self._retry_base_delay * (2**attempt),
                            self._retry_max_delay,
                        )
                    jitter = random.uniform(0, delay * 0.25)  # noqa: S311
                    delay += jitter

                    logger.warning(
                        "llm_transient_error_retrying",
                        attempt=attempt + 1,
                        max_retries=self._max_retries,
                        delay=delay,
                        error=str(exc),
                        rate_limited=is_rate_limit_error(exc),
                    )
                    if self._event_emitter:
                        self._event_emitter.emit_llm_retry(
                            task_id=task_id,
                            attempt=attempt + 1,
                            max_retries=self._max_retries,
                            error_message=str(exc),
                            delay_seconds=delay,
                        )
                    await asyncio.sleep(delay)
                    continue

                # Wrap transient errors that exhausted retries in LLMGatewayError
                # with a user-friendly message (permanent errors re-raise as-is)
                if is_transient_llm_error(exc):
                    raise self._wrap_llm_error(exc) from exc
                raise

        # Should not reach here, but satisfy type checker
        if last_error:
            raise self._wrap_llm_error(last_error)
        raise RuntimeError("unreachable")

    @staticmethod
    def _wrap_llm_error(exc: Exception) -> LLMGatewayError:
        """Wrap a raw LLM exception in LLMGatewayError with a clean message."""
        if is_rate_limit_error(exc):
            return LLMGatewayError(
                "Rate limited by the LLM provider. Please wait a moment and try again.",
                details={"original_error": str(exc), "error_type": "rate_limit"},
            )
        return LLMGatewayError(
            "LLM service is temporarily unavailable. Please try again.",
            details={"original_error": str(exc), "error_type": "transient"},
        )

    async def _do_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_text_chunk: Callable[[str], None] | None,
    ) -> LLMResponse:
        """Execute a single streaming chat completion call."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        text_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        stop_reason = "stop"
        input_tokens = 0
        output_tokens = 0

        async with await self._client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                if not chunk.choices and chunk.usage:
                    # Final usage-only chunk
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
                    continue

                for choice in chunk.choices:
                    delta = choice.delta

                    # Text content
                    if delta and delta.content:
                        text_parts.append(delta.content)
                        if on_text_chunk:
                            on_text_chunk(delta.content)

                    # Tool calls (accumulated across chunks)
                    if delta and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }
                            acc = tool_calls_acc[idx]
                            if tc_delta.id:
                                acc["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    acc["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    acc["arguments"] += tc_delta.function.arguments

                    # Finish reason
                    if choice.finish_reason:
                        stop_reason = choice.finish_reason

        # Parse accumulated tool calls
        parsed_tool_calls: list[ToolCallMessage] = []
        for idx in sorted(tool_calls_acc):
            acc = tool_calls_acc[idx]
            try:
                args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": acc["arguments"]}
            parsed_tool_calls.append(
                ToolCallMessage(
                    id=acc["id"],
                    name=acc["name"],
                    arguments=args,
                )
            )

        # Fallback: estimate tokens when the API doesn't provide usage data
        # (e.g. Anthropic's OpenAI-compatible endpoint ignores stream_options)
        if input_tokens == 0 and messages:
            input_tokens = sum(estimate_message_tokens(m) for m in messages)
        if output_tokens == 0 and (text_parts or parsed_tool_calls):
            output_tokens = estimate_tokens("".join(text_parts))
            for tc in parsed_tool_calls:
                output_tokens += estimate_tokens(tc.name)
                output_tokens += estimate_tokens(json.dumps(tc.arguments, default=str))

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=parsed_tool_calls,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
