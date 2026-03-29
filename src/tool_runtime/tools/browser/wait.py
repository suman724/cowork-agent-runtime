"""BrowserWait — wait for page conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from tool_runtime.exceptions import BrowserWaitTimeoutError
from tool_runtime.tools.browser.base_browser_tool import BaseBrowserTool

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

logger = structlog.get_logger(__name__)

_DEFAULT_TIMEOUT_MS = 30_000


class BrowserWaitTool(BaseBrowserTool):
    """Wait for a page condition before continuing."""

    @property
    def name(self) -> str:
        return "BrowserWait"

    @property
    def description(self) -> str:
        return (
            "Wait for a condition: an element to appear (CSS selector), "
            "a navigation to complete, or network to become idle."
        )

    @property
    def capability(self) -> str:
        return "Browser.Navigate"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "enum": ["element", "navigation", "networkidle"],
                    "description": "What to wait for.",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector (required when condition is 'element').",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30).",
                },
            },
            "required": ["condition"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ExecutionContext,  # noqa: ARG002
    ) -> RawToolOutput:
        self.validate_input(arguments)
        condition: str = arguments["condition"]
        selector: str | None = arguments.get("selector")
        timeout_seconds: int = arguments.get("timeout", 30)
        timeout_ms = timeout_seconds * 1000

        page = await self._get_page()

        try:
            if condition == "element":
                if not selector:
                    raise BrowserWaitTimeoutError(
                        "CSS selector is required when condition is 'element'"
                    )
                await page.wait_for_selector(selector, timeout=timeout_ms)
            elif condition == "navigation":
                await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            elif condition == "networkidle":
                await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except TimeoutError as exc:
            raise BrowserWaitTimeoutError(
                f"Wait condition '{condition}' not met within {timeout_seconds}s"
            ) from exc

        # Re-extract page state
        rendered = await self._extract_and_render(page)

        logger.info(
            "browser_wait_completed",
            condition=condition,
            selector=selector,
        )
        return self._page_state_output(rendered)
