"""Tests for core browser tools — Navigate, Click, Type, Select, Scroll, Back."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tool_runtime.exceptions import (
    BrowserAuthRequiredError,
    BrowserDomainApprovalRequiredError,
    BrowserDomainBlockedError,
    BrowserElementNotFoundError,
    BrowserNavigationError,
    BrowserSensitiveApprovalRequiredError,
)
from tool_runtime.models import ExecutionContext
from tool_runtime.tools.browser.back import BrowserBackTool
from tool_runtime.tools.browser.click import BrowserClickTool
from tool_runtime.tools.browser.navigate import BrowserNavigateTool, _domain_matches
from tool_runtime.tools.browser.scroll import BrowserScrollTool
from tool_runtime.tools.browser.select import BrowserSelectTool
from tool_runtime.tools.browser.type_text import BrowserTypeTool


def _make_browser_manager() -> MagicMock:
    """Create a mock BrowserManager."""
    mgr = MagicMock()
    mgr.approved_domains = set()

    page = AsyncMock()
    page.url = "https://example.com/page"
    page.title = AsyncMock(return_value="Example Page")
    page.screenshot = AsyncMock(return_value=b"\x89PNG_fake")

    # Mock accessibility snapshot — simple page with a button and text field
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(
        return_value={
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "heading", "name": "Welcome"},
                {"role": "textbox", "name": "Email"},
                {"role": "button", "name": "Submit"},
                {"role": "link", "name": "Home"},
            ],
        }
    )

    # Mock locators
    mock_locator = AsyncMock()
    mock_locator.first = AsyncMock()
    page.get_by_role = MagicMock(return_value=mock_locator)

    mgr.get_page = AsyncMock(return_value=page)
    return mgr


def _make_context(**kwargs: Any) -> ExecutionContext:
    return ExecutionContext(**kwargs)


# --- BrowserNavigate ---


class TestBrowserNavigate:
    def test_tool_properties(self) -> None:
        tool = BrowserNavigateTool(_make_browser_manager())
        assert tool.name == "BrowserNavigate"
        assert tool.capability == "Browser.Navigate"
        assert "url" in tool.input_schema["properties"]

    async def test_navigate_valid_url(self) -> None:
        mgr = _make_browser_manager()
        page = await mgr.get_page()
        page.goto = AsyncMock(return_value=MagicMock(status=200))
        mgr.approved_domains.add("example.com")

        tool = BrowserNavigateTool(mgr)
        result = await tool.execute({"url": "https://example.com"}, _make_context())
        assert result.output_text  # Should contain page state

    async def test_navigate_blocked_scheme(self) -> None:
        tool = BrowserNavigateTool(_make_browser_manager())
        with pytest.raises(BrowserNavigationError, match="Only http"):
            await tool.execute({"url": "file:///etc/passwd"}, _make_context())

    async def test_navigate_javascript_scheme_blocked(self) -> None:
        tool = BrowserNavigateTool(_make_browser_manager())
        with pytest.raises(BrowserNavigationError, match="Only http"):
            await tool.execute({"url": "javascript:alert(1)"}, _make_context())

    async def test_navigate_ssrf_localhost_blocked(self) -> None:
        tool = BrowserNavigateTool(_make_browser_manager())
        with pytest.raises(BrowserNavigationError, match="local/private"):
            await tool.execute({"url": "http://127.0.0.1/admin"}, _make_context())

    async def test_navigate_ssrf_private_ip_blocked(self) -> None:
        tool = BrowserNavigateTool(_make_browser_manager())
        with pytest.raises(BrowserNavigationError, match="local/private"):
            await tool.execute({"url": "http://10.0.0.1/internal"}, _make_context())

    async def test_navigate_blocked_domain(self) -> None:
        tool = BrowserNavigateTool(_make_browser_manager())
        ctx = _make_context(blocked_domains=["evil.com"])
        with pytest.raises(BrowserDomainBlockedError, match="blocked"):
            await tool.execute({"url": "https://evil.com"}, ctx)

    async def test_navigate_domain_not_in_allowlist(self) -> None:
        tool = BrowserNavigateTool(_make_browser_manager())
        ctx = _make_context(allowed_domains=["safe.com"])
        with pytest.raises(BrowserDomainBlockedError, match="not in"):
            await tool.execute({"url": "https://other.com"}, ctx)

    async def test_navigate_new_domain_triggers_approval(self) -> None:
        mgr = _make_browser_manager()
        tool = BrowserNavigateTool(mgr)
        # No pre-approved domains, no policy allowedDomains
        with pytest.raises(BrowserDomainApprovalRequiredError):
            await tool.execute({"url": "https://newsite.com"}, _make_context())

    async def test_navigate_approved_domain_skips_approval(self) -> None:
        mgr = _make_browser_manager()
        mgr.approved_domains.add("newsite.com")
        page = await mgr.get_page()
        page.goto = AsyncMock(return_value=MagicMock(status=200))

        tool = BrowserNavigateTool(mgr)
        result = await tool.execute({"url": "https://newsite.com"}, _make_context())
        assert result.output_text

    async def test_navigate_auth_401_detected(self) -> None:
        mgr = _make_browser_manager()
        mgr.approved_domains.add("secure.com")
        page = await mgr.get_page()
        page.goto = AsyncMock(return_value=MagicMock(status=401))

        tool = BrowserNavigateTool(mgr)
        with pytest.raises(BrowserAuthRequiredError, match="401"):
            await tool.execute({"url": "https://secure.com/api"}, _make_context())


# --- BrowserClick ---


class TestBrowserClick:
    def test_tool_properties(self) -> None:
        tool = BrowserClickTool(_make_browser_manager())
        assert tool.name == "BrowserClick"
        assert tool.capability == "Browser.Interact"

    async def test_click_valid_index(self) -> None:
        mgr = _make_browser_manager()
        tool = BrowserClickTool(mgr)
        result = await tool.execute({"index": 2}, _make_context())
        assert result.output_text  # Page state returned

    async def test_click_invalid_index(self) -> None:
        mgr = _make_browser_manager()
        tool = BrowserClickTool(mgr)
        with pytest.raises(BrowserElementNotFoundError, match="99"):
            await tool.execute({"index": 99}, _make_context())

    async def test_click_sensitive_destructive(self) -> None:
        mgr = _make_browser_manager()
        page = await mgr.get_page()
        # Override snapshot to include a destructive button
        page.accessibility.snapshot = AsyncMock(
            return_value={
                "role": "WebArea",
                "name": "Page",
                "children": [
                    {"role": "button", "name": "Delete project"},
                ],
            }
        )
        tool = BrowserClickTool(mgr)
        with pytest.raises(BrowserSensitiveApprovalRequiredError, match="destructive"):
            await tool.execute({"index": 1}, _make_context())


# --- BrowserType ---


class TestBrowserType:
    def test_tool_properties(self) -> None:
        tool = BrowserTypeTool(_make_browser_manager())
        assert tool.name == "BrowserType"
        assert tool.capability == "Browser.Interact"

    async def test_type_into_field(self) -> None:
        mgr = _make_browser_manager()
        tool = BrowserTypeTool(mgr)
        result = await tool.execute({"index": 1, "text": "test@example.com"}, _make_context())
        assert result.output_text

    async def test_type_password_field_sensitive(self) -> None:
        mgr = _make_browser_manager()
        page = await mgr.get_page()
        page.accessibility.snapshot = AsyncMock(
            return_value={
                "role": "WebArea",
                "name": "Page",
                "children": [
                    {"role": "textbox", "name": "Password", "inputType": "password"},
                ],
            }
        )
        tool = BrowserTypeTool(mgr)
        with pytest.raises(BrowserSensitiveApprovalRequiredError, match="password"):
            await tool.execute({"index": 1, "text": "secret"}, _make_context())


# --- BrowserSelect ---


class TestBrowserSelect:
    def test_tool_properties(self) -> None:
        tool = BrowserSelectTool(_make_browser_manager())
        assert tool.name == "BrowserSelect"

    async def test_select_option(self) -> None:
        mgr = _make_browser_manager()
        page = await mgr.get_page()
        page.accessibility.snapshot = AsyncMock(
            return_value={
                "role": "WebArea",
                "name": "Page",
                "children": [
                    {"role": "combobox", "name": "Country"},
                ],
            }
        )
        tool = BrowserSelectTool(mgr)
        result = await tool.execute({"index": 1, "value": "US"}, _make_context())
        assert result.output_text


# --- BrowserScroll ---


class TestBrowserScroll:
    def test_tool_properties(self) -> None:
        tool = BrowserScrollTool(_make_browser_manager())
        assert tool.name == "BrowserScroll"
        assert tool.capability == "Browser.Navigate"

    async def test_scroll_down(self) -> None:
        mgr = _make_browser_manager()
        tool = BrowserScrollTool(mgr)
        result = await tool.execute({"direction": "down"}, _make_context())
        assert result.output_text


# --- BrowserBack ---


class TestBrowserBack:
    def test_tool_properties(self) -> None:
        tool = BrowserBackTool(_make_browser_manager())
        assert tool.name == "BrowserBack"
        assert tool.capability == "Browser.Navigate"

    async def test_back_navigates(self) -> None:
        mgr = _make_browser_manager()
        tool = BrowserBackTool(mgr)
        result = await tool.execute({}, _make_context())
        assert result.output_text


# --- Domain matching ---


class TestDomainMatching:
    def test_exact_match(self) -> None:
        assert _domain_matches("example.com", "example.com")

    def test_exact_no_match(self) -> None:
        assert not _domain_matches("other.com", "example.com")

    def test_wildcard_subdomain(self) -> None:
        assert _domain_matches("sub.example.com", "*.example.com")

    def test_wildcard_nested_subdomain(self) -> None:
        assert _domain_matches("deep.sub.example.com", "*.example.com")

    def test_wildcard_base_domain(self) -> None:
        assert _domain_matches("example.com", "*.example.com")

    def test_wildcard_no_match(self) -> None:
        assert not _domain_matches("other.com", "*.example.com")
