"""MockLLMClient — configurable LLM responses for testing."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from agent_host.llm.models import LLMResponse, ToolCallMessage


class MockLLMClient:
    """Mock LLM client with a configurable response queue.

    Usage:
        mock = MockLLMClient()
        mock.enqueue(LLMResponse(text="Hello!", stop_reason="stop"))
        mock.enqueue(LLMResponse(
            text="",
            tool_calls=[ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/foo"})],
            stop_reason="tool_calls",
        ))
        mock.enqueue(LLMResponse(text="Done!", stop_reason="stop"))
    """

    def __init__(self) -> None:
        self._responses: deque[LLMResponse | Exception] = deque()
        self._call_log: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """Number of stream_chat calls made."""
        return len(self._call_log)

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Log of all stream_chat calls (messages, tools, task_id)."""
        return list(self._call_log)

    def enqueue(self, response: LLMResponse | Exception) -> None:
        """Add a response (or exception) to the queue."""
        self._responses.append(response)

    def enqueue_text(self, text: str, input_tokens: int = 10, output_tokens: int = 5) -> None:
        """Shorthand: enqueue a simple text response (natural termination)."""
        self._responses.append(
            LLMResponse(
                text=text,
                stop_reason="stop",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    def enqueue_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str = "tc-1",
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> None:
        """Shorthand: enqueue a response with a single tool call."""
        self._responses.append(
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCallMessage(
                        id=tool_call_id,
                        name=tool_name,
                        arguments=arguments or {},
                    )
                ],
                stop_reason="tool_calls",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        task_id: str = "",
        on_text_chunk: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Return the next queued response, firing text chunks if present."""
        self._call_log.append(
            {
                "messages": messages,
                "tools": tools,
                "task_id": task_id,
            }
        )

        if not self._responses:
            msg = "MockLLMClient: no more responses queued"
            raise RuntimeError(msg)

        response = self._responses.popleft()

        # If it's an exception, raise it
        if isinstance(response, Exception):
            raise response

        # Fire text chunks if callback provided
        if on_text_chunk and response.text:
            on_text_chunk(response.text)

        return response

    async def close(self) -> None:
        """No-op close."""
