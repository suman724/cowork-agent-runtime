"""Base class for all browser tools — shared BrowserManager and page state logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from tool_runtime.exceptions import BrowserElementNotFoundError
from tool_runtime.models import RawToolOutput
from tool_runtime.tools.base import BaseTool
from tool_runtime.tools.browser.dom_service import extract_page_snapshot
from tool_runtime.tools.browser.page_state import render_page_state

if TYPE_CHECKING:
    from playwright.async_api import Page

    from tool_runtime.models import ExecutionContext
    from tool_runtime.tools.browser.browser_manager import BrowserManager
    from tool_runtime.tools.browser.dom_service import IndexedElement, PageSnapshot

logger = structlog.get_logger(__name__)


class BaseBrowserTool(BaseTool):
    """Shared base for browser tools — provides page access and page state helpers."""

    def __init__(self, browser_manager: BrowserManager) -> None:
        self._browser_manager = browser_manager

    async def _get_page(self) -> Page:
        """Get the current browser page via BrowserManager."""
        return await self._browser_manager.get_page()

    async def _extract_and_render(self, page: Page) -> str:
        """Extract page state and render as markdown for LLM."""
        snapshot = await extract_page_snapshot(page)
        return render_page_state(snapshot)

    async def _extract_snapshot(self, page: Page) -> PageSnapshot:
        """Extract page snapshot for downstream use."""
        return await extract_page_snapshot(page)

    async def _capture_screenshot_base64(self, page: Page) -> str:
        """Capture a viewport screenshot and return base64-encoded PNG."""
        import base64

        screenshot_bytes = await page.screenshot(type="png")
        return base64.b64encode(screenshot_bytes).decode("ascii")

    def _emit_page_state(self, context: ExecutionContext, url: str, screenshot_b64: str) -> None:  # noqa: ARG002
        """Emit browser_page_state event for the Desktop App side panel."""
        if context.on_output_chunk:
            # We reuse on_output_chunk as the event emission channel.
            # The ToolExecutor wires this to EventEmitter.
            pass
        # Page state events are emitted by the BrowserManager callback,
        # not directly by tools. Tools capture screenshot and the manager
        # emits it. This method is a hook for future use.

    async def _resolve_element(self, page: Page, index: int) -> tuple[IndexedElement, PageSnapshot]:
        """Resolve an interactive element by index from the current page state.

        Raises BrowserElementNotFoundError if index is invalid.
        """
        snapshot = await self._extract_snapshot(page)
        element = snapshot.get_element(index)
        if element is None:
            raise BrowserElementNotFoundError(
                f"Element with index [{index}] not found. "
                f"Available indices: {[e.index for e in snapshot.elements]}"
            )
        return element, snapshot

    def _page_state_output(self, rendered: str) -> RawToolOutput:
        """Wrap rendered page state as a RawToolOutput."""
        return RawToolOutput(output_text=rendered)
