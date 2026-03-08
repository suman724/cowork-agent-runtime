"""Tests for MessageThread — conversation history management."""

from __future__ import annotations

from agent_host.llm.models import ToolCallMessage
from agent_host.thread.message_thread import MessageThread


class TestMessageThread:
    def test_init_with_system_prompt(self) -> None:
        thread = MessageThread(system_prompt="You are helpful.")
        assert thread.get_system_prompt() == "You are helpful."
        assert thread.message_count == 0

    def test_add_user_message(self) -> None:
        thread = MessageThread(system_prompt="test")
        thread.add_user_message("Hello")
        assert thread.message_count == 1
        assert thread.messages[0] == {"role": "user", "content": "Hello"}

    def test_add_assistant_message_text_only(self) -> None:
        thread = MessageThread(system_prompt="test")
        thread.add_assistant_message("Hi there!")
        assert thread.messages[0]["role"] == "assistant"
        assert thread.messages[0]["content"] == "Hi there!"
        assert "tool_calls" not in thread.messages[0]

    def test_add_assistant_message_with_tool_calls(self) -> None:
        thread = MessageThread(system_prompt="test")
        tool_calls = [ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/foo"})]
        thread.add_assistant_message("", tool_calls)
        msg = thread.messages[0]
        assert msg["role"] == "assistant"
        assert "content" not in msg  # content omitted when empty (API requires content OR tool_calls)
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "tc1"
        assert msg["tool_calls"][0]["function"]["name"] == "ReadFile"

    def test_add_assistant_message_empty_skipped(self) -> None:
        """Empty assistant messages (no text, no tool_calls) are silently dropped."""
        thread = MessageThread(system_prompt="test")
        thread.add_assistant_message("", None)
        assert thread.message_count == 0
        thread.add_assistant_message("", [])
        assert thread.message_count == 0

    def test_add_tool_result(self) -> None:
        thread = MessageThread(system_prompt="test")
        thread.add_tool_result("tc1", "ReadFile", '{"status": "success"}')
        msg = thread.messages[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc1"
        assert msg["name"] == "ReadFile"
        assert msg["content"] == '{"status": "success"}'

    def test_add_system_injection(self) -> None:
        thread = MessageThread(system_prompt="test")
        thread.add_system_injection("Working memory block")
        msg = thread.messages[0]
        assert msg["role"] == "system"
        assert msg["content"] == "Working memory block"

    def test_build_llm_payload_no_compactor(self) -> None:
        thread = MessageThread(system_prompt="System prompt")
        thread.add_user_message("Hello")
        thread.add_assistant_message("Hi!")
        payload = thread.build_llm_payload(100_000)
        assert len(payload) == 3
        assert payload[0] == {"role": "system", "content": "System prompt"}
        assert payload[1] == {"role": "user", "content": "Hello"}
        assert payload[2]["role"] == "assistant"

    def test_total_token_estimate(self) -> None:
        thread = MessageThread(system_prompt="short")
        thread.add_user_message("a" * 400)  # ~100 tokens
        total = thread.total_token_estimate()
        assert total > 100

    def test_set_system_prompt(self) -> None:
        thread = MessageThread(system_prompt="old")
        thread.set_system_prompt("new")
        assert thread.get_system_prompt() == "new"


class TestMessageThreadCheckpoint:
    def test_round_trip(self) -> None:
        thread = MessageThread(system_prompt="System prompt")
        thread.add_user_message("Hello")
        thread.add_assistant_message(
            "Hi!", [ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/x"})]
        )
        thread.add_tool_result("tc1", "ReadFile", "file content")
        thread.add_assistant_message("Done!")

        data = thread.to_checkpoint()
        restored = MessageThread.from_checkpoint(data)

        assert restored.get_system_prompt() == "System prompt"
        assert restored.message_count == 4
        assert restored.messages[0]["role"] == "user"
        assert restored.messages[1]["role"] == "assistant"
        assert restored.messages[2]["role"] == "tool"
        assert restored.messages[3]["role"] == "assistant"

    def test_empty_checkpoint(self) -> None:
        restored = MessageThread.from_checkpoint([])
        assert restored.get_system_prompt() == ""
        assert restored.message_count == 0

    def test_messages_are_copies(self) -> None:
        thread = MessageThread(system_prompt="test")
        thread.add_user_message("Hello")
        messages = thread.messages
        messages.append({"role": "user", "content": "Extra"})
        assert thread.message_count == 1  # original unchanged
