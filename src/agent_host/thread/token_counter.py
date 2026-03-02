"""Token estimation using character-count heuristic.

Same ~4 chars/token heuristic proven in the prior context_truncation.py.
"""

from __future__ import annotations

import json
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string (~4 characters per token)."""
    return max(1, len(text) // 4)


# Conservative flat estimate for image tokens. OpenAI's vision model uses ~85 tokens
# for low-res + ~170 tokens per 512px tile. We use ~1000 as a simple approximation.
_IMAGE_TOKEN_ESTIMATE = 1000


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate tokens for a complete OpenAI-format message.

    Accounts for role, content, tool_calls, tool_call_id fields, and images.
    Adds overhead per message (~4 tokens for formatting).
    """
    total = 4  # per-message overhead (role, delimiters)

    content = message.get("content")
    if isinstance(content, str) and content:
        total += estimate_tokens(content)
    elif isinstance(content, list):
        # Multimodal content blocks
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif block.get("type") == "image_url":
                    total += _IMAGE_TOKEN_ESTIMATE

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
