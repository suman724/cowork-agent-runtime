"""DomService — accessibility tree extraction with indexed interactive elements.

Extracts a structured representation of the current page for the LLM,
using Playwright's accessibility API. Interactive elements are assigned
monotonic indices so the LLM can reference them by number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from playwright.async_api import Page

logger = structlog.get_logger(__name__)

# Roles that indicate interactive elements in the accessibility tree
_INTERACTIVE_ROLES = frozenset(
    {
        "link",
        "button",
        "textbox",
        "searchbox",
        "combobox",
        "listbox",
        "option",
        "checkbox",
        "radio",
        "switch",
        "slider",
        "spinbutton",
        "tab",
        "menuitem",
        "menuitemcheckbox",
        "menuitemradio",
        "treeitem",
    }
)


@dataclass
class IndexedElement:
    """An interactive element with a stable index for LLM reference."""

    index: int
    role: str
    name: str
    tag: str = ""
    value: str = ""
    checked: bool | None = None
    disabled: bool = False
    focused: bool = False
    description: str = ""
    # Raw accessibility node for downstream use (sensitive detection, etc.)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class PageSnapshot:
    """Complete snapshot of the current page state."""

    url: str
    title: str
    elements: list[IndexedElement]
    content_text: str  # Markdown-like text content for LLM context

    def get_element(self, index: int) -> IndexedElement | None:
        """Look up an element by its index."""
        for el in self.elements:
            if el.index == index:
                return el
        return None


async def extract_page_snapshot(page: Page) -> PageSnapshot:
    """Extract a complete page snapshot with indexed interactive elements.

    Uses Playwright's accessibility tree for structured content and
    assigns monotonic indices to interactive elements.
    """
    url = page.url
    title = await page.title()

    try:
        a11y_tree = await page.accessibility.snapshot()  # type: ignore[attr-defined]
    except Exception:
        logger.warning("a11y_snapshot_failed", url=url, exc_info=True)
        a11y_tree = None

    elements: list[IndexedElement] = []
    content_parts: list[str] = []
    index_counter = [1]  # Mutable counter for closure

    if a11y_tree:
        _walk_tree(a11y_tree, elements, content_parts, index_counter, depth=0)

    content_text = "\n".join(content_parts)

    logger.debug(
        "page_snapshot_extracted",
        url=url,
        element_count=len(elements),
        content_length=len(content_text),
    )

    return PageSnapshot(
        url=url,
        title=title,
        elements=elements,
        content_text=content_text,
    )


def _walk_tree(
    node: dict[str, Any],
    elements: list[IndexedElement],
    content_parts: list[str],
    index_counter: list[int],
    depth: int,
) -> None:
    """Depth-first walk of the accessibility tree.

    Interactive elements get indexed. All elements contribute to content text.
    """
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    is_interactive = role in _INTERACTIVE_ROLES

    if is_interactive and name:
        idx = index_counter[0]
        index_counter[0] += 1

        el = IndexedElement(
            index=idx,
            role=role,
            name=name,
            value=str(value) if value else "",
            checked=node.get("checked"),
            disabled=node.get("disabled", False),
            focused=node.get("focused", False),
            description=node.get("description", ""),
            raw=node,
        )
        elements.append(el)

        # Render with index marker
        label = _format_element_label(el)
        content_parts.append(f"[{idx}] {label}")

    elif name and role not in ("generic", "none", "presentation"):
        # Non-interactive content — render as context
        text = _format_content_node(role, name, depth)
        if text:
            content_parts.append(text)

    # Recurse into children
    for child in node.get("children", []):
        _walk_tree(child, elements, content_parts, index_counter, depth + 1)


def _format_element_label(el: IndexedElement) -> str:
    """Format an interactive element for LLM display."""
    parts = [el.role]
    if el.name:
        parts.append(f'"{el.name}"')
    if el.value:
        parts.append(f"value={el.value}")
    if el.checked is not None:
        parts.append("checked" if el.checked else "unchecked")
    if el.disabled:
        parts.append("(disabled)")
    return " ".join(parts)


def _format_content_node(role: str, name: str, depth: int) -> str:
    """Format a non-interactive content node for LLM context."""
    if role == "heading":
        # Approximate heading level from depth
        level = min(depth + 1, 6)
        return f"{'#' * level} {name}"
    if role in ("paragraph", "text", "StaticText"):
        return name
    if role == "img":
        return f"[Image: {name}]" if name else ""
    if role == "separator":
        return "---"
    if role in ("list", "listitem"):
        return f"- {name}" if name else ""
    if role == "table":
        return f"[Table: {name}]" if name else "[Table]"
    # Default: include if it has meaningful text
    if name and len(name) > 1:
        return name
    return ""
