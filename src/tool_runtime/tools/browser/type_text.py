"""BrowserType — type text into an input field."""

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


class BrowserTypeTool(BaseBrowserTool):
    """Type text into an input field by its index."""

    @property
    def name(self) -> str:
        return "BrowserType"

    @property
    def description(self) -> str:
        return (
            "Type text into an input field by its index number. "
            "By default clears the field first. Returns updated page state."
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
                    "description": "Index of the input field.",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the field.",
                },
                "clearFirst": {
                    "type": "boolean",
                    "description": "Clear existing content before typing (default: true).",
                },
                "pressEnter": {
                    "type": "boolean",
                    "description": "Press Enter after typing (default: false).",
                },
            },
            "required": ["index", "text"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> RawToolOutput:  # noqa: ARG002
        self.validate_input(arguments)
        index: int = arguments["index"]
        text: str = arguments["text"]
        clear_first: bool = arguments.get("clearFirst", True)
        press_enter: bool = arguments.get("pressEnter", False)

        page = await self._get_page()

        # 1. Resolve element
        element, _snapshot = await self._resolve_element(page, index)

        # 2. Sensitive field check (Tier 2)
        sensitivity = detect_sensitive(element)
        if sensitivity is not None:
            raise BrowserSensitiveApprovalRequiredError(
                action_summary=(
                    f"Type into [{element.index}] {element.role} "
                    f'"{element.name}" ({sensitivity.value} field detected)'
                ),
            )

        # 3. Type into the field
        try:
            locator = page.get_by_role(element.role, name=element.name)  # type: ignore[arg-type]
            if clear_first:
                await locator.first.fill(text)
            else:
                await locator.first.type(text)

            if press_enter:
                await locator.first.press("Enter")
        except Exception as exc:
            raise BrowserElementNotInteractableError(
                f'Could not type into [{index}] {element.role} "{element.name}": {exc}'
            ) from exc

        # 4. Wait for DOM settlement
        try:  # noqa: SIM105
            await page.wait_for_timeout(500)
        except Exception:  # noqa: S110
            pass

        # 5. Re-extract page state
        rendered = await self._extract_and_render(page)

        logger.info(
            "browser_typed",
            index=index,
            role=element.role,
            name=element.name,
            text_length=len(text),
        )
        return self._page_state_output(rendered)
