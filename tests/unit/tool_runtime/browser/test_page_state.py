"""Tests for PageState — token-efficient page rendering for LLM."""

from __future__ import annotations

from tool_runtime.tools.browser.dom_service import IndexedElement, PageSnapshot
from tool_runtime.tools.browser.page_state import render_page_state


def _make_snapshot(
    url: str = "https://example.com",
    title: str = "Example",
    elements: list[IndexedElement] | None = None,
    content: str = "",
) -> PageSnapshot:
    return PageSnapshot(
        url=url,
        title=title,
        elements=elements or [],
        content_text=content,
    )


class TestRenderPageState:
    def test_includes_url_and_title(self) -> None:
        snapshot = _make_snapshot(url="https://github.com", title="GitHub")
        rendered = render_page_state(snapshot)
        assert "GitHub" in rendered
        assert "https://github.com" in rendered

    def test_includes_elements_with_indices(self) -> None:
        elements = [
            IndexedElement(index=1, role="link", name="Home"),
            IndexedElement(index=2, role="button", name="Submit"),
        ]
        rendered = render_page_state(_make_snapshot(elements=elements))
        assert '[1] link "Home"' in rendered
        assert '[2] button "Submit"' in rendered

    def test_includes_element_value(self) -> None:
        elements = [
            IndexedElement(index=1, role="textbox", name="Email", value="user@test.com"),
        ]
        rendered = render_page_state(_make_snapshot(elements=elements))
        assert "(value: user@test.com)" in rendered

    def test_includes_checked_state(self) -> None:
        elements = [
            IndexedElement(index=1, role="checkbox", name="Agree", checked=True),
            IndexedElement(index=2, role="checkbox", name="Newsletter", checked=False),
        ]
        rendered = render_page_state(_make_snapshot(elements=elements))
        assert "[checked]" in rendered
        assert "[unchecked]" in rendered

    def test_includes_disabled_state(self) -> None:
        elements = [
            IndexedElement(index=1, role="button", name="Save", disabled=True),
        ]
        rendered = render_page_state(_make_snapshot(elements=elements))
        assert "(disabled)" in rendered

    def test_includes_content(self) -> None:
        rendered = render_page_state(_make_snapshot(content="Hello world"))
        assert "Hello world" in rendered

    def test_truncation_within_budget(self) -> None:
        # Small page — no truncation
        rendered = render_page_state(_make_snapshot(content="short"), max_tokens=1000)
        assert "truncated" not in rendered

    def test_truncation_exceeds_budget(self) -> None:
        # Very long content with tiny budget
        long_content = "x" * 10000
        rendered = render_page_state(_make_snapshot(content=long_content), max_tokens=100)
        assert "truncated" in rendered
        # Should be within budget (100 tokens * 4 chars + some overhead)
        assert len(rendered) < 1000

    def test_empty_page(self) -> None:
        rendered = render_page_state(_make_snapshot())
        assert "Page:" in rendered
        assert "Interactive Elements" not in rendered  # No elements section
