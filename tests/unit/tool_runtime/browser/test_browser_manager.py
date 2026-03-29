"""Tests for BrowserManager — lifecycle, idle timeout, pause/resume, crash detection."""

from __future__ import annotations

import asyncio
import importlib
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tool_runtime.exceptions import BrowserLaunchError
from tool_runtime.tools.browser.browser_manager import (
    BrowserManager,
    BrowserState,
)

_has_playwright = importlib.util.find_spec("playwright") is not None


def _make_manager(
    workspace_dir: str = "/tmp/test-workspace",
    idle_timeout: int = 600,
    **kwargs: Any,
) -> BrowserManager:
    return BrowserManager(
        workspace_dir=workspace_dir,
        idle_timeout_seconds=idle_timeout,
        **kwargs,
    )


def _mock_playwright_context() -> tuple[AsyncMock, AsyncMock, MagicMock]:
    """Create mock Playwright, context, and page objects."""
    mock_page = MagicMock()
    mock_page.is_closed.return_value = False

    mock_context = AsyncMock()
    mock_context.pages = [mock_page]
    mock_context.close = AsyncMock()
    mock_context.on = MagicMock()

    mock_pw = AsyncMock()
    mock_pw.chromium.launch_persistent_context = AsyncMock(return_value=mock_context)
    mock_pw.stop = AsyncMock()

    return mock_pw, mock_context, mock_page


@pytest.mark.skipif(not _has_playwright, reason="playwright not installed")
class TestBrowserManagerLifecycle:
    def test_initial_state_is_idle(self) -> None:
        mgr = _make_manager()
        assert mgr.state == BrowserState.IDLE

    def test_profile_dir(self) -> None:
        mgr = _make_manager(workspace_dir="/home/user/project")
        assert mgr.profile_dir == "/home/user/project/.cowork/browser-profile"

    @patch("playwright.async_api.async_playwright")
    async def test_get_page_launches_browser(self, mock_async_pw: MagicMock) -> None:
        mock_pw, _mock_ctx, mock_page = _mock_playwright_context()
        mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw)

        mgr = _make_manager()
        page = await mgr.get_page()

        assert page is mock_page
        assert mgr.state == BrowserState.ACTIVE
        mock_pw.chromium.launch_persistent_context.assert_called_once()

        await mgr.close()

    @patch("playwright.async_api.async_playwright")
    async def test_get_page_reuses_existing(self, mock_async_pw: MagicMock) -> None:
        mock_pw, _mock_ctx, _mock_page = _mock_playwright_context()
        mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw)

        mgr = _make_manager()
        page1 = await mgr.get_page()
        page2 = await mgr.get_page()

        assert page1 is page2
        # Should only launch once
        mock_pw.chromium.launch_persistent_context.assert_called_once()

        await mgr.close()

    @patch("playwright.async_api.async_playwright")
    async def test_close_transitions_to_idle(self, mock_async_pw: MagicMock) -> None:
        mock_pw, mock_ctx, _mock_page = _mock_playwright_context()
        mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw)

        mgr = _make_manager()
        await mgr.get_page()
        assert mgr.state == BrowserState.ACTIVE

        await mgr.close()
        assert mgr.state == BrowserState.IDLE
        mock_ctx.close.assert_called_once()
        mock_pw.stop.assert_called_once()

    @patch("playwright.async_api.async_playwright")
    async def test_close_emits_event(self, mock_async_pw: MagicMock) -> None:
        mock_pw, _mock_ctx, _mock_page = _mock_playwright_context()
        mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw)

        events: list[tuple[str, dict[str, Any]]] = []
        mgr = _make_manager(on_event=lambda et, p: events.append((et, p)))
        await mgr.get_page()
        events.clear()

        await mgr.close()
        assert any(e[0] == "browser_stopped" for e in events)

    @patch("playwright.async_api.async_playwright")
    async def test_launch_emits_browser_started(self, mock_async_pw: MagicMock) -> None:
        mock_pw, _mock_ctx, _mock_page = _mock_playwright_context()
        mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw)

        events: list[tuple[str, dict[str, Any]]] = []
        mgr = _make_manager(on_event=lambda et, p: events.append((et, p)))
        await mgr.get_page()

        assert any(e[0] == "browser_started" for e in events)
        await mgr.close()


@pytest.mark.skipif(not _has_playwright, reason="playwright not installed")
class TestBrowserManagerCrashDetection:
    def test_crash_transitions_to_suspended(self) -> None:
        mgr = _make_manager()
        mgr._state = BrowserState.ACTIVE
        mgr._on_browser_disconnected()
        assert mgr.state == BrowserState.SUSPENDED

    def test_crash_during_shutdown_ignored(self) -> None:
        mgr = _make_manager()
        mgr._state = BrowserState.SHUTTING_DOWN
        mgr._on_browser_disconnected()
        assert mgr.state == BrowserState.SHUTTING_DOWN

    def test_repeated_crashes_raise(self) -> None:
        mgr = _make_manager()
        now = time.monotonic()
        mgr._crash_timestamps = [now - 60, now - 30, now - 10]

        with pytest.raises(BrowserLaunchError, match="crashed 3 times"):
            mgr._check_crash_limit()

    def test_old_crashes_expire(self) -> None:
        mgr = _make_manager()
        old = time.monotonic() - 600  # 10 min ago — outside 5-min window
        mgr._crash_timestamps = [old, old + 1, old + 2]
        # Should not raise — all crashes are old
        mgr._check_crash_limit()

    def test_crash_emits_event(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        mgr = _make_manager(on_event=lambda et, p: events.append((et, p)))
        mgr._state = BrowserState.ACTIVE
        mgr._on_browser_disconnected()

        assert any(e[0] == "browser_stopped" and e[1]["reason"] == "crashed" for e in events)

    @patch("playwright.async_api.async_playwright")
    async def test_resume_after_crash_relaunches(self, mock_async_pw: MagicMock) -> None:
        mock_pw, _mock_ctx, mock_page = _mock_playwright_context()
        mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw)

        mgr = _make_manager()
        mgr._state = BrowserState.SUSPENDED  # Simulate post-crash state

        page = await mgr.get_page()
        assert page is mock_page
        assert mgr.state == BrowserState.ACTIVE

        await mgr.close()


@pytest.mark.skipif(not _has_playwright, reason="playwright not installed")
class TestBrowserManagerPauseResume:
    @patch("playwright.async_api.async_playwright")
    async def test_pause_blocks_get_page(self, mock_async_pw: MagicMock) -> None:
        mock_pw, _mock_ctx, mock_page = _mock_playwright_context()
        mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw)

        mgr = _make_manager()
        await mgr.get_page()  # Launch first

        mgr.pause()
        assert not mgr._pause_event.is_set()

        # get_page should block — use wait_for to prove it
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(mgr.get_page(), timeout=0.1)

        mgr.resume()
        assert mgr._pause_event.is_set()

        # Now should succeed
        page = await asyncio.wait_for(mgr.get_page(), timeout=1.0)
        assert page is mock_page

        await mgr.close()

    def test_pause_emits_takeover_started(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        mgr = _make_manager(on_event=lambda et, p: events.append((et, p)))
        mgr.pause()
        assert any(e[0] == "browser_takeover_started" for e in events)

    def test_resume_emits_takeover_ended(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        mgr = _make_manager(on_event=lambda et, p: events.append((et, p)))
        mgr.pause()
        events.clear()
        mgr.resume()
        assert any(e[0] == "browser_takeover_ended" for e in events)


class TestBrowserManagerDomains:
    def test_approved_domains_starts_empty(self) -> None:
        mgr = _make_manager()
        assert mgr.approved_domains == set()

    def test_approved_domains_persists(self) -> None:
        mgr = _make_manager()
        mgr.approved_domains.add("github.com")
        mgr.approved_domains.add("jira.atlassian.net")
        assert "github.com" in mgr.approved_domains
        assert "jira.atlassian.net" in mgr.approved_domains
