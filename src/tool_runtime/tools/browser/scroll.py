"""BrowserScroll — scroll the page in a direction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from tool_runtime.tools.browser.base_browser_tool import BaseBrowserTool

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

logger = structlog.get_logger(__name__)

_SCROLL_AMOUNTS = {
    "page": "window.innerHeight",
    "half": "window.innerHeight / 2",
    "line": "100",
}

_LAZY_LOAD_WAIT_MS = 500


class BrowserScrollTool(BaseBrowserTool):
    """Scroll the page up, down, left, or right."""

    @property
    def name(self) -> str:
        return "BrowserScroll"

    @property
    def description(self) -> str:
        return (
            "Scroll the page in a direction. Returns updated page state "
            "after scrolling (lazy-loaded content will be included)."
        )

    @property
    def capability(self) -> str:
        return "Browser.Navigate"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Scroll direction.",
                },
                "amount": {
                    "type": "string",
                    "enum": ["page", "half", "line"],
                    "description": "Scroll amount (default: page).",
                },
            },
            "required": ["direction"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> RawToolOutput:  # noqa: ARG002
        self.validate_input(arguments)
        direction: str = arguments["direction"]
        amount: str = arguments.get("amount", "page")

        page = await self._get_page()

        # Build scroll JS
        distance_expr = _SCROLL_AMOUNTS.get(amount, "window.innerHeight")
        if direction == "down":
            js = f"window.scrollBy(0, {distance_expr})"
        elif direction == "up":
            js = f"window.scrollBy(0, -({distance_expr}))"
        elif direction == "right":
            js = f"window.scrollBy({distance_expr}, 0)"
        else:  # left
            js = f"window.scrollBy(-({distance_expr}), 0)"

        await page.evaluate(js)

        # Wait for lazy-loaded content
        await page.wait_for_timeout(_LAZY_LOAD_WAIT_MS)

        # Re-extract page state
        rendered = await self._extract_and_render(page)

        logger.info("browser_scrolled", direction=direction, amount=amount)
        return self._page_state_output(rendered)
