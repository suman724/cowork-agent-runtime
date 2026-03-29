"""PageState — token-efficient page representation for the LLM.

Renders a PageSnapshot into markdown with indexed interactive elements,
respecting a token budget to avoid bloating the LLM context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tool_runtime.tools.browser.dom_service import PageSnapshot

# Approximate chars-per-token ratio for English text
_CHARS_PER_TOKEN = 4
_DEFAULT_MAX_TOKENS = 20_000


def render_page_state(
    snapshot: PageSnapshot,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """Render a page snapshot as markdown for LLM consumption.

    Includes URL, title, interactive elements with indices, and
    page content. Truncates with 80/20 head/tail if exceeding budget.
    """
    parts: list[str] = []

    # Header
    parts.append(f"## Page: {snapshot.title}")
    parts.append(f"URL: {snapshot.url}")
    parts.append("")

    # Interactive elements summary
    if snapshot.elements:
        parts.append("### Interactive Elements")
        for el in snapshot.elements:
            label = f"[{el.index}] {el.role}"
            if el.name:
                label += f' "{el.name}"'
            if el.value:
                label += f" (value: {el.value})"
            if el.checked is not None:
                label += " [checked]" if el.checked else " [unchecked]"
            if el.disabled:
                label += " (disabled)"
            parts.append(label)
        parts.append("")

    # Page content
    if snapshot.content_text:
        parts.append("### Page Content")
        parts.append(snapshot.content_text)

    rendered = "\n".join(parts)

    # Truncate if exceeding token budget
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(rendered) > max_chars:
        rendered = _truncate_head_tail(rendered, max_chars)

    return rendered


def _truncate_head_tail(text: str, max_chars: int) -> str:
    """Keep first 80% and last 20% with a marker in between."""
    head_size = int(max_chars * 0.8)
    tail_size = int(max_chars * 0.2)
    removed = len(text) - head_size - tail_size
    return (
        text[:head_size] + f"\n\n... [{removed} characters truncated] ...\n\n" + text[-tail_size:]
    )
