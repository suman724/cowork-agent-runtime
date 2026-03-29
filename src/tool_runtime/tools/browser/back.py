"""BrowserBack — navigate browser history back."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from tool_runtime.exceptions import BrowserNavigationError
from tool_runtime.tools.browser.base_browser_tool import BaseBrowserTool

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

logger = structlog.get_logger(__name__)


class BrowserBackTool(BaseBrowserTool):
    """Navigate the browser back in history."""

    @property
    def name(self) -> str:
        return "BrowserBack"

    @property
    def description(self) -> str:
        return "Go back to the previous page in browser history. Returns updated page state."

    @property
    def capability(self) -> str:
        return "Browser.Navigate"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> RawToolOutput:  # noqa: ARG002
        self.validate_input(arguments)

        page = await self._get_page()

        try:
            await page.go_back(wait_until="domcontentloaded")
        except Exception as exc:
            raise BrowserNavigationError(f"Failed to go back: {exc}") from exc

        # Re-extract page state
        rendered = await self._extract_and_render(page)

        logger.info("browser_back", url=page.url)
        return self._page_state_output(rendered)
