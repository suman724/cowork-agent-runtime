"""Tests for token_counter — token estimation heuristics."""

from __future__ import annotations

from agent_host.thread.token_counter import estimate_message_tokens, estimate_tokens


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 1  # minimum 1

    def test_short_string(self) -> None:
        assert estimate_tokens("hi") == 1  # 2 chars // 4 = 0, but min 1

    def test_normal_string(self) -> None:
        # 100 chars -> ~25 tokens
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_long_string(self) -> None:
        text = "x" * 4000
        assert estimate_tokens(text) == 1000


class TestEstimateMessageTokens:
    def test_simple_user_message(self) -> None:
        msg = {"role": "user", "content": "Hello world"}
        tokens = estimate_message_tokens(msg)
        assert tokens >= 4  # at least overhead

    def test_message_with_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "function": {"name": "ReadFile", "arguments": '{"path": "/foo/bar.py"}'},
                }
            ],
        }
        tokens = estimate_message_tokens(msg)
        assert tokens > 4  # overhead + tool call name + arguments

    def test_tool_result_message(self) -> None:
        msg = {
            "role": "tool",
            "tool_call_id": "tc1",
            "name": "ReadFile",
            "content": "file contents here" * 10,
        }
        tokens = estimate_message_tokens(msg)
        assert tokens > 4

    def test_empty_content(self) -> None:
        msg = {"role": "system", "content": ""}
        tokens = estimate_message_tokens(msg)
        assert tokens == 4  # just overhead

    def test_dict_arguments(self) -> None:
        """Tool call with dict arguments (not string) should be serialized."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "function": {"name": "ReadFile", "arguments": {"path": "/foo"}},
                }
            ],
        }
        tokens = estimate_message_tokens(msg)
        assert tokens > 4
