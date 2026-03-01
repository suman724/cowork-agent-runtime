"""Context window truncation — prevent overflow by dropping oldest messages."""

from __future__ import annotations

import json
from typing import Any


def estimate_tokens(text: str) -> int:
    """Rough token estimate using chars/4 heuristic."""
    return max(1, len(text) // 4)


def estimate_contents_tokens(contents: list[Any]) -> int:
    """Estimate total tokens across all Content parts."""
    total = 0
    for content in contents:
        for part in getattr(content, "parts", []) or []:
            if hasattr(part, "text") and part.text:
                total += estimate_tokens(part.text)
            elif hasattr(part, "function_call") and part.function_call:
                total += estimate_tokens(
                    json.dumps(getattr(part.function_call, "args", {}), default=str)
                )
            elif hasattr(part, "function_response") and part.function_response:
                total += estimate_tokens(
                    json.dumps(getattr(part.function_response, "response", {}), default=str)
                )
    return total


def truncate_contents(
    contents: list[Any],
    max_tokens: int,
    recency_window: int = 4,
) -> list[Any]:
    """Drop oldest messages (keeping first + last recency_window) to fit token budget.

    Args:
        contents: List of ADK Content objects.
        max_tokens: Maximum estimated token count.
        recency_window: Number of most-recent messages to always keep.

    Returns:
        Truncated contents list (may be the same object if no truncation needed).
    """
    if not contents or estimate_contents_tokens(contents) <= max_tokens:
        return contents

    # Keep first message (system/initial prompt) + last recency_window messages
    keep_head = contents[:1]
    if len(contents) <= recency_window + 1:
        keep_tail = contents[1:]
        middle: list[Any] = []
    else:
        keep_tail = contents[-recency_window:]
        middle = list(contents[1:-recency_window])

    while middle and estimate_contents_tokens(keep_head + middle + keep_tail) > max_tokens:
        middle.pop(0)  # drop oldest

    return keep_head + middle + keep_tail
