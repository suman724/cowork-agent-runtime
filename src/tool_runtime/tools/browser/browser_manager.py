"""BrowserManager — Playwright browser lifecycle management.

Manages a single Playwright persistent browser context with states:
Idle → Launching → Active → Suspended → ShuttingDown.

Lazy launch: no browser process until first get_page() call.
Idle timeout: browser closes after configurable inactivity period.
Crash recovery: re-launches on next call with same profile directory.
User takeover: pause/resume via asyncio Event.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

from tool_runtime.exceptions import BrowserCrashedError, BrowserLaunchError

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import BrowserContext, Page, Playwright

logger = structlog.get_logger(__name__)

_MAX_CRASHES_WINDOW = 5 * 60  # 5 minutes
_MAX_CRASHES_COUNT = 3


class BrowserState(Enum):
    """Browser lifecycle states."""

    IDLE = "idle"
    LAUNCHING = "launching"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    SHUTTING_DOWN = "shutting_down"


class BrowserManager:
    """Manages a single Playwright browser instance per session.

    The browser is launched lazily on first get_page() call and closed
    after an idle timeout or session end. State persists via Chrome's
    userDataDir (profile directory) — no custom serialization needed.
    """

    def __init__(
        self,
        workspace_dir: str,
        idle_timeout_seconds: int = 600,
        headless: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._workspace_dir = workspace_dir
        self._idle_timeout = idle_timeout_seconds
        self._headless = headless
        self._viewport = {"width": viewport_width, "height": viewport_height}
        self._on_event = on_event

        self._state = BrowserState.IDLE
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._idle_timer: asyncio.TimerHandle | None = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default

        # Crash tracking
        self._crash_timestamps: list[float] = []

        # Approved domains for this session (Tier 1 domain approval)
        self.approved_domains: set[str] = set()

    @property
    def state(self) -> BrowserState:
        """Current browser lifecycle state."""
        return self._state

    @property
    def profile_dir(self) -> str:
        """Path to Chrome's persistent profile directory."""
        from pathlib import Path

        return str(Path(self._workspace_dir) / ".cowork" / "browser-profile")

    async def get_page(self) -> Page:
        """Get the current browser page, launching if needed.

        This is the primary entry point for all browser tools.
        Handles lazy launch, resume from suspension, and pause gate.
        """
        # Wait if paused (user takeover)
        await self._pause_event.wait()

        if self._state in (BrowserState.IDLE, BrowserState.SUSPENDED):
            await self._launch()
        elif self._state == BrowserState.SHUTTING_DOWN:
            raise BrowserLaunchError("Browser is shutting down")
        elif self._state == BrowserState.LAUNCHING:
            # Another call is already launching — wait for it
            while self._state == BrowserState.LAUNCHING:
                await asyncio.sleep(0.05)
            if self._page is None:
                raise BrowserLaunchError("Browser failed to launch")

        self._reset_idle_timer()

        if self._page is None or self._page.is_closed():
            raise BrowserCrashedError("Browser page is not available")

        return self._page

    async def close(self) -> None:
        """Close the browser and clean up resources."""
        if self._state == BrowserState.SHUTTING_DOWN:
            return

        prev_state = self._state
        self._state = BrowserState.SHUTTING_DOWN
        self._cancel_idle_timer()

        logger.info("browser_closing", previous_state=prev_state.value)

        try:
            if self._context is not None:
                await self._context.close()
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception:
            logger.warning("browser_close_error", exc_info=True)
        finally:
            self._context = None
            self._page = None
            self._playwright = None
            self._state = BrowserState.IDLE

        self._emit_event("browser_stopped", {"reason": "closed"})

    async def _launch(self) -> None:
        """Launch Playwright with a persistent browser context."""
        self._check_crash_limit()
        self._state = BrowserState.LAUNCHING

        logger.info(
            "browser_launching",
            profile_dir=self.profile_dir,
            headless=self._headless,
            viewport=self._viewport,
        )

        try:
            from pathlib import Path

            from playwright.async_api import async_playwright

            # Ensure profile directory exists
            Path(self.profile_dir).mkdir(parents=True, exist_ok=True)

            self._playwright = await async_playwright().start()
            from playwright.async_api import ViewportSize

            viewport: ViewportSize = {
                "width": self._viewport["width"],
                "height": self._viewport["height"],
            }
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=self.profile_dir,
                headless=self._headless,
                viewport=viewport,
                args=["--disk-cache-size=0"],
            )

            # Use the default page or create one
            if self._context.pages:
                self._page = self._context.pages[0]
            else:
                self._page = await self._context.new_page()

            # Crash detection — "close" event passes BrowserContext
            self._context.on("close", lambda _ctx: self._on_browser_disconnected())

            self._state = BrowserState.ACTIVE
            self._reset_idle_timer()

            logger.info("browser_launched", state=self._state.value)
            self._emit_event("browser_started", {"browserChannel": "chromium"})

        except Exception as exc:
            self._state = BrowserState.IDLE
            self._playwright = None
            self._context = None
            self._page = None
            logger.warning("browser_launch_failed", error=str(exc), exc_info=True)
            raise BrowserLaunchError(f"Failed to launch browser: {exc}") from exc

    def _on_browser_disconnected(self) -> None:
        """Handle unexpected browser disconnection (crash)."""
        if self._state == BrowserState.SHUTTING_DOWN:
            return  # Expected — we're closing

        logger.warning("browser_crashed", previous_state=self._state.value)
        self._crash_timestamps.append(time.monotonic())
        self._state = BrowserState.SUSPENDED
        self._context = None
        self._page = None
        self._cancel_idle_timer()

        self._emit_event("browser_stopped", {"reason": "crashed"})

    def _check_crash_limit(self) -> None:
        """Raise if too many crashes in the tracking window."""
        now = time.monotonic()
        self._crash_timestamps = [
            t for t in self._crash_timestamps if now - t < _MAX_CRASHES_WINDOW
        ]
        if len(self._crash_timestamps) >= _MAX_CRASHES_COUNT:
            raise BrowserLaunchError(
                f"Browser crashed {_MAX_CRASHES_COUNT} times in "
                f"{_MAX_CRASHES_WINDOW // 60} minutes. Please restart the session."
            )

    # --- Idle timeout ---

    def _reset_idle_timer(self) -> None:
        """Reset the idle timeout. Called after every tool interaction."""
        self._cancel_idle_timer()
        loop = asyncio.get_event_loop()
        self._idle_timer = loop.call_later(
            self._idle_timeout, lambda: asyncio.ensure_future(self._on_idle_timeout())
        )

    def _cancel_idle_timer(self) -> None:
        """Cancel the pending idle timer."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    async def _on_idle_timeout(self) -> None:
        """Handle idle timeout — suspend the browser."""
        if self._state != BrowserState.ACTIVE:
            return

        logger.info("browser_idle_timeout", timeout_seconds=self._idle_timeout)
        self._state = BrowserState.SUSPENDED

        try:
            if self._context is not None:
                await self._context.close()
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception:
            logger.warning("browser_suspend_error", exc_info=True)
        finally:
            self._context = None
            self._page = None
            self._playwright = None

        self._emit_event("browser_stopped", {"reason": "idle"})

    # --- User takeover (pause / resume) ---

    def pause(self) -> None:
        """Pause the browser — blocks get_page() until resume()."""
        self._pause_event.clear()
        logger.info("browser_paused")
        self._emit_event("browser_takeover_started", {})

    def resume(self) -> None:
        """Resume the browser — unblocks get_page()."""
        self._pause_event.set()
        logger.info("browser_resumed")
        self._emit_event("browser_takeover_ended", {})

    # --- Event emission ---

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit a browser lifecycle event via the registered callback."""
        if self._on_event:
            try:
                self._on_event(event_type, payload)
            except Exception:
                logger.warning(
                    "browser_event_emission_failed",
                    event_type=event_type,
                    exc_info=True,
                )
