"""BrowserExtract — read page content in various formats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from tool_runtime.models import RawToolOutput
from tool_runtime.tools.browser.base_browser_tool import BaseBrowserTool

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext

logger = structlog.get_logger(__name__)


class BrowserExtractTool(BaseBrowserTool):
    """Extract page content as markdown, text, or HTML."""

    @property
    def name(self) -> str:
        return "BrowserExtract"

    @property
    def description(self) -> str:
        return (
            "Extract page content as markdown (default), text, or HTML. "
            "Optionally scope to a CSS selector. Read-only — no visual change."
        )

    @property
    def capability(self) -> str:
        return "Browser.Extract"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector to scope extraction (optional).",
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "text", "html"],
                    "description": "Output format (default: markdown).",
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
        selector: str | None = arguments.get("selector")
        fmt: str = arguments.get("format", "markdown")

        page = await self._get_page()

        if selector:
            # Scoped extraction
            try:
                element = page.locator(selector)
                if fmt == "html":
                    content = await element.first.inner_html()
                elif fmt == "text":
                    content = await element.first.inner_text()
                else:
                    content = await element.first.inner_text()
            except Exception as exc:
                content = f"Could not extract from selector '{selector}': {exc}"
        elif fmt == "html":
            content = await page.content()
        elif fmt == "text":
            content = await page.inner_text("body")
        else:
            # Markdown — use page state renderer
            rendered = await self._extract_and_render(page)
            return RawToolOutput(output_text=rendered)

        logger.info("browser_extracted", format=fmt, selector=selector)
        return RawToolOutput(output_text=content)
