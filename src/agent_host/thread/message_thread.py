"""MessageThread — ordered conversation history in OpenAI message format."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent_host.thread.token_counter import estimate_message_tokens

if TYPE_CHECKING:
    from agent_host.llm.models import ToolCallMessage
    from agent_host.thread.compactor import ContextCompactor


class MessageThread:
    """Manages the ordered list of conversation messages for the agent loop.

    All messages are stored in OpenAI chat completion format:
    - system: ``{"role": "system", "content": "..."}``
    - user: ``{"role": "user", "content": "..."}``
    - assistant: ``{"role": "assistant", "content": "...", "tool_calls": [...]}``
    - tool: ``{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}``
    """

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._messages: list[dict[str, Any]] = []

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return a copy of the message list (excluding system prompt)."""
        return list(self._messages)

    @property
    def message_count(self) -> int:
        """Return the number of messages (excluding system prompt)."""
        return len(self._messages)

    def get_system_prompt(self) -> str:
        """Return the system prompt text."""
        return self._system_prompt

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt."""
        self._system_prompt = prompt

    def add_user_message(self, content: str) -> None:
        """Append a user message."""
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self, text: str, tool_calls: list[ToolCallMessage] | None = None
    ) -> None:
        """Append an assistant message, optionally with tool calls.

        Skips empty messages (no text AND no tool_calls) to avoid producing
        invalid API payloads — the OpenAI chat completion format requires
        assistant messages to have at least one of content or tool_calls.
        """
        has_text = bool(text)
        has_tool_calls = bool(tool_calls)

        if not has_text and not has_tool_calls:
            return  # Skip empty assistant messages

        msg: dict[str, Any] = {"role": "assistant"}

        if has_text:
            msg["content"] = text

        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, default=str),
                    },
                }
                for tc in tool_calls
            ]

        self._messages.append(msg)

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        image_url: str | None = None,
    ) -> None:
        """Append a tool result message.

        When image_url is provided (data URI), the content is sent as multimodal
        content blocks so the LLM can see the image.
        """
        if image_url:
            content: str | list[dict[str, Any]] = [
                {"type": "text", "text": result},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            content = result

        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": content,
            }
        )

    def add_system_injection(self, content: str) -> None:
        """Append a system injection message (for working memory, etc.)."""
        self._messages.append({"role": "system", "content": content})

    def build_llm_payload(
        self,
        max_tokens: int,
        compactor: ContextCompactor | None = None,
    ) -> list[dict[str, Any]]:
        """Build the message list for an LLM API call.

        Returns [system_prompt] + messages, optionally compacted to fit budget.
        """
        system_msg: dict[str, Any] = {"role": "system", "content": self._system_prompt}
        all_messages = [system_msg, *self._messages]

        if compactor:
            return compactor.compact(all_messages, max_tokens)
        return all_messages

    def total_token_estimate(self) -> int:
        """Estimate total tokens across all messages including system prompt."""
        system_msg: dict[str, Any] = {"role": "system", "content": self._system_prompt}
        total = estimate_message_tokens(system_msg)
        for msg in self._messages:
            total += estimate_message_tokens(msg)
        return total

    def to_checkpoint(self) -> list[dict[str, Any]]:
        """Serialize thread state for checkpoint persistence."""
        return [
            {"system_prompt": self._system_prompt},
            *self._messages,
        ]

    @classmethod
    def from_checkpoint(cls, data: list[dict[str, Any]]) -> MessageThread:
        """Restore a MessageThread from checkpoint data."""
        if not data:
            return cls(system_prompt="")

        first = data[0]
        system_prompt = first.get("system_prompt", "")
        thread = cls(system_prompt=system_prompt)
        thread._messages = list(data[1:])
        return thread
