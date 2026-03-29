"""BrowserClick — click an interactive element on the page."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from tool_runtime.exceptions import (
    BrowserElementNotInteractableError,
    BrowserSensitiveApprovalRequiredError,
)
from tool_runtime.tools.browser.base_browser_tool import BaseBrowserTool
from tool_runtime.tools.browser.sensitive_detector import detect_sensitive

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

logger = structlog.get_logger(__name__)

_SETTLEMENT_TIMEOUT_MS = 5000


class BrowserClickTool(BaseBrowserTool):
    """Click an interactive element by its index."""

    @property
    def name(self) -> str:
        return "BrowserClick"

    @property
    def description(self) -> str:
        return (
            "Click an interactive element on the page by its index number. "
            "Returns the updated page state after clicking."
        )

    @property
    def capability(self) -> str:
        return "Browser.Interact"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Index of the element to click.",
                },
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "description": "Mouse button (default: left).",
                },
            },
            "required": ["index"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> RawToolOutput:  # noqa: ARG002
        self.validate_input(arguments)
        index: int = arguments["index"]
        button: str = arguments.get("button", "left")

        page = await self._get_page()

        # 1. Resolve element by index
        element, _snapshot = await self._resolve_element(page, index)

        # 2. Sensitive element check (Tier 2)
        sensitivity = detect_sensitive(element)
        if sensitivity is not None:
            screenshot_b64 = await self._capture_screenshot_base64(page)
            raise BrowserSensitiveApprovalRequiredError(
                action_summary=(
                    f"Click [{element.index}] {element.role} "
                    f'"{element.name}" ({sensitivity.value} detected)'
                ),
                screenshot_base64=screenshot_b64,
            )

        # 3. Click the element via Playwright locator
        try:
            # Use role-based locator with name for precision
            locator = page.get_by_role(element.role, name=element.name)  # type: ignore[arg-type]
            await locator.first.scroll_into_view_if_needed()
            await locator.first.click(button=button)  # type: ignore[arg-type]
        except Exception as exc:
            raise BrowserElementNotInteractableError(
                f'Could not click [{index}] {element.role} "{element.name}": {exc}'
            ) from exc

        # 4. Wait for DOM settlement
        try:  # noqa: SIM105
            await page.wait_for_load_state("domcontentloaded", timeout=_SETTLEMENT_TIMEOUT_MS)
        except Exception:  # noqa: S110
            pass  # Best effort — page may not navigate

        # 5. Re-extract page state
        rendered = await self._extract_and_render(page)

        logger.info(
            "browser_clicked",
            index=index,
            role=element.role,
            name=element.name,
        )
        return self._page_state_output(rendered)
