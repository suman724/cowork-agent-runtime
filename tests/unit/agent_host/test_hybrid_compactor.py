"""Tests for HybridCompactor — observation masking + LLM summarization."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from agent_host.thread.compactor import HybridCompactor


def _tool_msg(name: str, content: dict) -> dict:
    return {"role": "tool", "name": name, "content": json.dumps(content)}


def _assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _system_msg(text: str) -> dict:
    return {"role": "system", "content": text}


class TestObservationMasking:
    def test_masks_old_tool_results(self) -> None:
        """Tool results outside recency window should be masked."""
        compactor = HybridCompactor(recency_window=2)
        messages = [
            _system_msg("System prompt"),
            _tool_msg("ReadFile", {"status": "success", "output": "x" * 1000}),
            _assistant_msg("Got it"),
            _tool_msg("WriteFile", {"status": "success", "output": "written"}),
            _assistant_msg("Done"),
            # recency window = last 2
            _tool_msg("ReadFile", {"status": "success", "output": "recent"}),
            _assistant_msg("Final"),
        ]

        masked = compactor._mask_old_observations(messages)

        # Old tool results (index 1, 3) should be masked
        assert masked[1]["content"].startswith("[ReadFile:")
        assert masked[3]["content"].startswith("[WriteFile:")
        # Recent tool result (index 5) should be preserved
        assert "recent" in masked[5]["content"]
        # System prompt and assistant messages should be unchanged
        assert masked[0]["content"] == "System prompt"
        assert masked[2]["content"] == "Got it"

    def test_preserves_recent_tool_results(self) -> None:
        """Tool results within recency window should not be masked."""
        compactor = HybridCompactor(recency_window=10)
        messages = [
            _system_msg("System"),
            _tool_msg("ReadFile", {"status": "success", "output": "data"}),
            _assistant_msg("Response"),
        ]
        masked = compactor._mask_old_observations(messages)
        # All within recency window — nothing masked
        assert json.loads(masked[1]["content"])["output"] == "data"

    def test_masks_failed_tool(self) -> None:
        """Failed tool results should show error in mask."""
        compactor = HybridCompactor(recency_window=1)
        messages = [
            _system_msg("System"),
            _tool_msg(
                "RunCommand",
                {"status": "failed", "error": {"message": "Permission denied"}},
            ),
            _assistant_msg("Recent"),
        ]
        masked = compactor._mask_old_observations(messages)
        assert "failed" in masked[1]["content"]
        assert "Permission denied" in masked[1]["content"]

    def test_masks_denied_tool(self) -> None:
        """Denied tool results should be marked as denied."""
        compactor = HybridCompactor(recency_window=1)
        messages = [
            _system_msg("System"),
            _tool_msg("WriteFile", {"status": "denied"}),
            _assistant_msg("Recent"),
        ]
        masked = compactor._mask_old_observations(messages)
        assert "denied" in masked[1]["content"]


class TestHybridCompactFitsBudget:
    def test_masking_sufficient(self) -> None:
        """When masking alone fits budget, no further compaction needed."""
        compactor = HybridCompactor(recency_window=2)
        messages = [
            _system_msg("System"),
            _tool_msg("ReadFile", {"status": "success", "output": "x" * 500}),
            _assistant_msg("Got it"),
            _assistant_msg("Recent 1"),
            _assistant_msg("Recent 2"),
        ]
        result = compactor.compact(messages, budget_tokens=10_000)
        # Should return masked version without dropping anything
        assert len(result) == 5

    def test_falls_back_to_drop_oldest(self) -> None:
        """When masking and no cached summary, should fall back to drop-oldest."""
        compactor = HybridCompactor(recency_window=2)
        messages = [
            _system_msg("System"),
        ]
        # Add many messages to exceed a tiny budget
        for i in range(20):
            messages.append(_assistant_msg(f"Message {i} " + "y" * 100))

        result = compactor.compact(messages, budget_tokens=50)
        # Should have compacted — fewer messages
        assert len(result) < len(messages)

    def test_mask_only_mode(self) -> None:
        """mask_only=True should never trigger LLM summarization."""
        compactor = HybridCompactor(recency_window=2, mask_only=True)
        # precompute_summary should be a no-op
        assert compactor._mask_only is True


class TestCachedSummary:
    def test_apply_cached_summary(self) -> None:
        """Cached summary should replace old messages."""
        compactor = HybridCompactor(recency_window=2)
        compactor._cached_summary = "Summary of prior work"

        messages = [
            _system_msg("System"),
            _assistant_msg("Old 1"),
            _assistant_msg("Old 2"),
            _assistant_msg("Recent 1"),
            _assistant_msg("Recent 2"),
        ]

        result = compactor._apply_cached_summary(messages, budget_tokens=10_000)
        assert result is not None
        # Should have: system + summary + 2 recent
        assert len(result) == 4
        assert "Summary of prior work" in result[1]["content"]

    def test_apply_cached_summary_exceeds_budget(self) -> None:
        """If summary + recency window still exceeds budget, return None."""
        compactor = HybridCompactor(recency_window=2)
        compactor._cached_summary = "Summary " * 100

        messages = [
            _system_msg("System"),
            _assistant_msg("Recent 1"),
            _assistant_msg("Recent 2"),
        ]

        result = compactor._apply_cached_summary(messages, budget_tokens=1)
        assert result is None


class TestPrecomputeSummary:
    async def test_precompute_calls_llm(self) -> None:
        """precompute_summary should call LLM when masking isn't enough."""
        compactor = HybridCompactor(recency_window=2)

        # Build messages that exceed a tiny budget even after masking
        messages = [_system_msg("System")]
        for i in range(10):
            messages.append(_assistant_msg(f"Content {i} " + "x" * 200))

        mock_llm = MagicMock()
        mock_llm.stream_chat = AsyncMock(return_value=MagicMock(text="Summary of conversation"))

        await compactor.precompute_summary(messages, budget_tokens=10, llm_client=mock_llm)
        assert compactor._cached_summary == "Summary of conversation"
        mock_llm.stream_chat.assert_awaited_once()

    async def test_precompute_skipped_when_fits(self) -> None:
        """precompute_summary should skip LLM call when masking fits budget."""
        compactor = HybridCompactor(recency_window=2)

        messages = [
            _system_msg("System"),
            _assistant_msg("Short"),
        ]

        mock_llm = MagicMock()
        mock_llm.stream_chat = AsyncMock()

        await compactor.precompute_summary(messages, budget_tokens=10_000, llm_client=mock_llm)
        mock_llm.stream_chat.assert_not_awaited()
        assert compactor._cached_summary is None

    async def test_precompute_skipped_mask_only(self) -> None:
        """precompute_summary should be no-op when mask_only=True."""
        compactor = HybridCompactor(recency_window=2, mask_only=True)

        messages = [_system_msg("System")] + [_assistant_msg("x" * 500) for _ in range(20)]

        mock_llm = MagicMock()
        mock_llm.stream_chat = AsyncMock()

        await compactor.precompute_summary(messages, budget_tokens=10, llm_client=mock_llm)
        mock_llm.stream_chat.assert_not_awaited()
