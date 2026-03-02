"""Token estimation using character-count heuristic.

Same ~4 chars/token heuristic proven in the prior context_truncation.py.
"""

from __future__ import annotations

import json
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string (~4 characters per token)."""
    return max(1, len(text) // 4)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate tokens for a complete OpenAI-format message.

    Accounts for role, content, tool_calls, and tool_call_id fields.
    Adds overhead per message (~4 tokens for formatting).
    """
    total = 4  # per-message overhead (role, delimiters)

    content = message.get("content")
    if isinstance(content, str) and content:
        total += estimate_tokens(content)

    # Tool calls (assistant messages with function calls)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            fn = tc.get("function", {})
            total += estimate_tokens(fn.get("name", ""))
            args = fn.get("arguments", "")
            if isinstance(args, str):
                total += estimate_tokens(args)
            else:
                total += estimate_tokens(json.dumps(args, default=str))

    # Tool result messages
    tool_call_id = message.get("tool_call_id")
    if isinstance(tool_call_id, str):
        total += estimate_tokens(tool_call_id)

    return total
