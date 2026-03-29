"""Tests for DomService — accessibility tree extraction and element indexing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from tool_runtime.tools.browser.dom_service import (
    IndexedElement,
    PageSnapshot,
    extract_page_snapshot,
)


def _make_page(a11y_tree: dict | None = None, url: str = "https://example.com") -> MagicMock:
    """Create a mock Playwright page with accessibility snapshot."""
    page = MagicMock()
    page.url = url
    page.title = AsyncMock(return_value="Example Page")
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(return_value=a11y_tree)
    return page


class TestExtractPageSnapshot:
    async def test_empty_page(self) -> None:
        page = _make_page(a11y_tree=None)
        snapshot = await extract_page_snapshot(page)
        assert snapshot.url == "https://example.com"
        assert snapshot.title == "Example Page"
        assert snapshot.elements == []

    async def test_indexes_interactive_elements(self) -> None:
        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "link", "name": "Home"},
                {"role": "button", "name": "Submit"},
                {"role": "textbox", "name": "Email"},
            ],
        }
        page = _make_page(a11y_tree=tree)
        snapshot = await extract_page_snapshot(page)

        assert len(snapshot.elements) == 3
        assert snapshot.elements[0].index == 1
        assert snapshot.elements[0].role == "link"
        assert snapshot.elements[0].name == "Home"
        assert snapshot.elements[1].index == 2
        assert snapshot.elements[1].role == "button"
        assert snapshot.elements[2].index == 3
        assert snapshot.elements[2].role == "textbox"

    async def test_skips_non_interactive(self) -> None:
        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "heading", "name": "Welcome"},
                {"role": "paragraph", "name": "Some text here"},
                {"role": "button", "name": "Click me"},
            ],
        }
        page = _make_page(a11y_tree=tree)
        snapshot = await extract_page_snapshot(page)

        # Only the button should be indexed
        assert len(snapshot.elements) == 1
        assert snapshot.elements[0].name == "Click me"
        # But heading and paragraph should appear in content
        assert "Welcome" in snapshot.content_text
        assert "Some text here" in snapshot.content_text

    async def test_depth_first_ordering(self) -> None:
        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {
                    "role": "navigation",
                    "name": "Nav",
                    "children": [
                        {"role": "link", "name": "First"},
                        {"role": "link", "name": "Second"},
                    ],
                },
                {"role": "button", "name": "Third"},
            ],
        }
        page = _make_page(a11y_tree=tree)
        snapshot = await extract_page_snapshot(page)

        assert len(snapshot.elements) == 3
        assert snapshot.elements[0].name == "First"
        assert snapshot.elements[1].name == "Second"
        assert snapshot.elements[2].name == "Third"

    async def test_checkbox_state(self) -> None:
        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "checkbox", "name": "Agree to terms", "checked": True},
            ],
        }
        page = _make_page(a11y_tree=tree)
        snapshot = await extract_page_snapshot(page)

        assert snapshot.elements[0].checked is True

    async def test_content_includes_headings(self) -> None:
        tree = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "heading", "name": "Main Title"},
            ],
        }
        page = _make_page(a11y_tree=tree)
        snapshot = await extract_page_snapshot(page)

        assert "# Main Title" in snapshot.content_text


class TestPageSnapshotLookup:
    def test_get_element_found(self) -> None:
        el = IndexedElement(index=5, role="button", name="Submit")
        snapshot = PageSnapshot(url="https://x.com", title="X", elements=[el], content_text="")
        assert snapshot.get_element(5) is el

    def test_get_element_not_found(self) -> None:
        snapshot = PageSnapshot(url="https://x.com", title="X", elements=[], content_text="")
        assert snapshot.get_element(99) is None
