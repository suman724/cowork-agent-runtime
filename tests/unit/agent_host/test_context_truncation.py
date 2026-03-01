"""Tests for context window truncation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_host.agent.context_truncation import (
    estimate_contents_tokens,
    estimate_tokens,
    truncate_contents,
)


@dataclass
class FakePart:
    text: str | None = None
    function_call: Any = None
    function_response: Any = None


@dataclass
class FakeContent:
    parts: list[FakePart] = field(default_factory=list)


def _make_content(text: str) -> FakeContent:
    return FakeContent(parts=[FakePart(text=text)])


class TestEstimateTokens:
    def test_basic_estimation(self) -> None:
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcdefgh") == 2

    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 1  # max(1, 0)

    def test_long_string(self) -> None:
        text = "a" * 400
        assert estimate_tokens(text) == 100


class TestEstimateContentsTokens:
    def test_single_content(self) -> None:
        contents = [_make_content("a" * 100)]
        assert estimate_contents_tokens(contents) == 25

    def test_multiple_contents(self) -> None:
        contents = [_make_content("a" * 100), _make_content("b" * 200)]
        assert estimate_contents_tokens(contents) == 75

    def test_empty_contents(self) -> None:
        assert estimate_contents_tokens([]) == 0


class TestTruncateContents:
    def test_no_truncation_under_limit(self) -> None:
        contents = [_make_content("short")]
        result = truncate_contents(contents, max_tokens=1000)
        assert result is contents  # same object, unchanged

    def test_truncation_drops_oldest(self) -> None:
        # Create 10 messages, each with 100 chars = ~25 tokens = ~250 total
        contents = [_make_content(f"msg-{i} " + "x" * 93) for i in range(10)]
        # Set budget to ~125 tokens = ~5 messages worth
        result = truncate_contents(contents, max_tokens=125, recency_window=3)
        # Should keep first + last 3, drop middle until fits
        assert result[0] is contents[0]  # first preserved
        assert result[-1] is contents[-1]  # last preserved
        assert result[-2] is contents[-2]
        assert result[-3] is contents[-3]
        assert len(result) <= 10

    def test_keeps_first_and_last(self) -> None:
        contents = [_make_content("x" * 400) for _ in range(20)]
        result = truncate_contents(contents, max_tokens=200, recency_window=4)
        # First message always kept
        assert result[0] is contents[0]
        # Last 4 always kept
        assert result[-1] is contents[-1]
        assert result[-2] is contents[-2]
        assert result[-3] is contents[-3]
        assert result[-4] is contents[-4]

    def test_empty_contents(self) -> None:
        assert truncate_contents([], max_tokens=100) == []

    def test_single_message_over_limit(self) -> None:
        # Even one message over limit, we still return at least head+tail
        contents = [_make_content("x" * 10000)]
        result = truncate_contents(contents, max_tokens=10)
        assert len(result) == 1  # can't drop the only message

    def test_few_messages_within_recency_window(self) -> None:
        # Only 3 messages with recency_window=4 — no truncation possible
        contents = [_make_content("x" * 40) for _ in range(3)]
        result = truncate_contents(contents, max_tokens=5, recency_window=4)
        # Can't drop anything, all within head+tail
        assert len(result) == 3
