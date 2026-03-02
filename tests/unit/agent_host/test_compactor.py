"""Tests for ContextCompactor — drop-oldest compaction strategy."""

from __future__ import annotations

from typing import Any

from agent_host.thread.compactor import DropOldestCompactor


def _make_msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


class TestDropOldestCompactor:
    def test_no_compaction_under_budget(self) -> None:
        compactor = DropOldestCompactor(recency_window=4)
        messages = [
            _make_msg("system", "short"),
            _make_msg("user", "hi"),
            _make_msg("assistant", "hello"),
        ]
        result = compactor.compact(messages, budget_tokens=100_000)
        assert len(result) == 3

    def test_drops_oldest_middle_messages(self) -> None:
        compactor = DropOldestCompactor(recency_window=2)
        # System prompt + 10 messages, each with ~100 tokens
        messages = [_make_msg("system", "System " + "x" * 100)]
        for i in range(10):
            messages.append(_make_msg("user", f"Message {i} " + "y" * 100))

        # Budget that fits system + marker + ~2 recent
        result = compactor.compact(messages, budget_tokens=150)
        # Should have: system + marker + some middle + 2 recent
        assert len(result) < len(messages)
        assert result[0]["role"] == "system"

    def test_marker_message_inserted(self) -> None:
        compactor = DropOldestCompactor(recency_window=2)
        messages = [_make_msg("system", "s")]
        for _ in range(20):
            messages.append(_make_msg("user", "x" * 200))

        result = compactor.compact(messages, budget_tokens=200)
        # Look for the marker message
        markers = [m for m in result if "earlier messages omitted" in m.get("content", "")]
        assert len(markers) == 1
        assert markers[0]["role"] == "system"

    def test_preserves_recency_window(self) -> None:
        compactor = DropOldestCompactor(recency_window=3)
        messages = [_make_msg("system", "s")]
        for i in range(10):
            messages.append(_make_msg("user", f"msg-{i} " + "a" * 200))

        result = compactor.compact(messages, budget_tokens=300)
        # Last 3 messages should always be preserved
        tail = result[-3:]
        assert "msg-7" in tail[0]["content"]
        assert "msg-8" in tail[1]["content"]
        assert "msg-9" in tail[2]["content"]

    def test_empty_messages(self) -> None:
        compactor = DropOldestCompactor()
        assert compactor.compact([], 1000) == []

    def test_small_thread_no_middle(self) -> None:
        """If thread is smaller than recency_window + 1, no middle exists."""
        compactor = DropOldestCompactor(recency_window=10)
        messages = [
            _make_msg("system", "s"),
            _make_msg("user", "q"),
            _make_msg("assistant", "a"),
        ]
        result = compactor.compact(messages, budget_tokens=100_000)
        assert len(result) == 3

    def test_system_prompt_always_kept(self) -> None:
        compactor = DropOldestCompactor(recency_window=1)
        messages = [_make_msg("system", "SYSTEM")]
        for _ in range(20):
            messages.append(_make_msg("user", "x" * 200))

        result = compactor.compact(messages, budget_tokens=200)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "SYSTEM"


class TestLongThreadCompaction:
    """Tests with 100+ messages to verify compaction at scale."""

    def test_100_message_thread(self) -> None:
        """100+ message thread should compact to fit budget."""
        compactor = DropOldestCompactor(recency_window=20)
        messages = [_make_msg("system", "System prompt " + "x" * 200)]
        for i in range(120):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append(_make_msg(role, f"Turn {i}: " + "y" * 200))

        # Budget that forces significant compaction
        result = compactor.compact(messages, budget_tokens=2000)
        assert len(result) < len(messages)
        # System prompt preserved
        assert result[0]["role"] == "system"
        # Last 20 messages preserved
        tail_20 = result[-20:]
        for msg in tail_20:
            assert msg["role"] in {"user", "assistant"}
        # Marker exists
        markers = [m for m in result if "earlier messages omitted" in m.get("content", "")]
        assert len(markers) == 1

    def test_recency_window_respected_with_many_messages(self) -> None:
        """Recency window must be preserved even with heavy compaction."""
        compactor = DropOldestCompactor(recency_window=10)
        messages = [_make_msg("system", "s")]
        for i in range(150):
            messages.append(_make_msg("user", f"msg-{i} " + "a" * 100))

        # Very tight budget — should still keep last 10
        result = compactor.compact(messages, budget_tokens=500)
        tail_10 = result[-10:]
        for idx, msg in enumerate(tail_10):
            expected_num = 140 + idx
            assert f"msg-{expected_num}" in msg["content"]

    def test_token_budget_respected(self) -> None:
        """Result should fit within token budget."""
        from agent_host.thread.token_counter import estimate_message_tokens

        compactor = DropOldestCompactor(recency_window=5)
        messages = [_make_msg("system", "s" * 100)]
        for i in range(100):
            messages.append(_make_msg("user", f"msg-{i} " + "z" * 400))

        budget = 1000
        result = compactor.compact(messages, budget_tokens=budget)
        total = sum(estimate_message_tokens(m) for m in result)
        assert total <= budget

    def test_alternating_roles_preserved(self) -> None:
        """Compacted thread should maintain valid role alternation in tail."""
        compactor = DropOldestCompactor(recency_window=6)
        messages = [_make_msg("system", "s")]
        for i in range(100):
            if i % 3 == 0:
                messages.append(_make_msg("user", f"user-{i} " + "a" * 100))
            elif i % 3 == 1:
                messages.append(_make_msg("assistant", f"asst-{i} " + "b" * 100))
            else:
                messages.append(
                    {"role": "tool", "tool_call_id": f"tc-{i}", "name": "ReadFile", "content": "ok"}
                )

        result = compactor.compact(messages, budget_tokens=500)
        # Tail should be preserved exactly
        assert result[-6:] == messages[-6:]
