"""BrowserScreenshot — capture viewport or full page as PNG."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import structlog

from tool_runtime.models import ImageContent, RawToolOutput
from tool_runtime.tools.browser.base_browser_tool import BaseBrowserTool

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext

logger = structlog.get_logger(__name__)


class BrowserScreenshotTool(BaseBrowserTool):
    """Capture a screenshot of the current page."""

    @property
    def name(self) -> str:
        return "BrowserScreenshot"

    @property
    def description(self) -> str:
        return (
            "Capture a screenshot of the current page viewport (or full page). "
            "Returns the image for visual understanding of complex layouts."
        )

    @property
    def capability(self) -> str:
        return "Browser.Extract"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fullPage": {
                    "type": "boolean",
                    "description": "Capture the full scrollable page (default: false).",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector to screenshot a specific element.",
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ExecutionContext,  # noqa: ARG002
    ) -> RawToolOutput:
        self.validate_input(arguments)
        full_page: bool = arguments.get("fullPage", False)
        selector: str | None = arguments.get("selector")

        page = await self._get_page()

        if selector:
            element = page.locator(selector)
            screenshot_bytes = await element.first.screenshot(type="png")
        else:
            screenshot_bytes = await page.screenshot(type="png", full_page=full_page)

        b64_data = base64.b64encode(screenshot_bytes).decode("ascii")

        logger.info(
            "browser_screenshot",
            full_page=full_page,
            selector=selector,
            size_bytes=len(screenshot_bytes),
        )

        return RawToolOutput(
            output_text=f"Screenshot captured ({len(screenshot_bytes)} bytes)",
            image_content=ImageContent(
                media_type="image/png",
                base64_data=b64_data,
            ),
        )
