"""Output truncation for tool results.

When tool output exceeds max_bytes: keep first 80% (head) + last 20% (tail)
with a marker between indicating how many bytes were removed.
"""

from __future__ import annotations

from tool_runtime.models import DEFAULT_MAX_OUTPUT_BYTES


def truncate_output(text: str, max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> str:
    """Truncate text output preserving head and tail.

    Returns the original text if it fits within max_bytes.
    Otherwise returns 80% head + marker + 20% tail, with the marker
    size accounted for in the byte budget.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text

    # Estimate marker size to reserve space for it within the budget.
    # The marker is: "\n\n[... truncated {N} bytes ...]\n\n"
    # where N can be up to len(str(len(encoded))) digits.
    marker_overhead = len(f"\n\n[... truncated {len(encoded)} bytes ...]\n\n".encode())
    content_budget = max(0, max_bytes - marker_overhead)

    head_bytes = int(content_budget * 0.8)
    tail_bytes = content_budget - head_bytes
    truncated_bytes = len(encoded) - head_bytes - tail_bytes

    # Decode with replacement to handle splits within multi-byte chars
    head = encoded[:head_bytes].decode("utf-8", errors="replace")
    tail = encoded[-tail_bytes:].decode("utf-8", errors="replace") if tail_bytes > 0 else ""
    marker = f"\n\n[... truncated {truncated_bytes} bytes ...]\n\n"

    return head + marker + tail
