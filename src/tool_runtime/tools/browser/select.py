"""BrowserSelect — select options in dropdowns, checkboxes, and radio buttons."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from tool_runtime.exceptions import BrowserElementNotInteractableError
from tool_runtime.tools.browser.base_browser_tool import BaseBrowserTool

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

logger = structlog.get_logger(__name__)


class BrowserSelectTool(BaseBrowserTool):
    """Select an option in a dropdown, checkbox, or radio button."""

    @property
    def name(self) -> str:
        return "BrowserSelect"

    @property
    def description(self) -> str:
        return (
            "Select an option in a dropdown, checkbox, or radio button by element index. "
            "For dropdowns, provide the value to select. For checkboxes/radio, toggles the state."
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
                    "description": "Index of the form element.",
                },
                "value": {
                    "type": "string",
                    "description": "Value to select (for dropdowns). "
                    "For checkboxes/radio, omit to toggle.",
                },
            },
            "required": ["index"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> RawToolOutput:  # noqa: ARG002
        self.validate_input(arguments)
        index: int = arguments["index"]
        value: str | None = arguments.get("value")

        page = await self._get_page()

        # 1. Resolve element
        element, _snapshot = await self._resolve_element(page, index)

        # 2. Select based on element type
        try:
            locator = page.get_by_role(element.role, name=element.name)  # type: ignore[arg-type]

            if element.role == "combobox" or element.role == "listbox":
                # Dropdown — select by value or label
                if value:
                    await locator.first.select_option(value)
                else:
                    raise BrowserElementNotInteractableError(
                        f"Dropdown [{index}] requires a 'value' argument"
                    )
            elif element.role in ("checkbox", "switch"):
                # Checkbox/switch — toggle via click
                await locator.first.click()
            elif element.role == "radio":
                # Radio — select via click
                await locator.first.click()
            else:
                # Fallback — try click
                await locator.first.click()

        except BrowserElementNotInteractableError:
            raise
        except Exception as exc:
            raise BrowserElementNotInteractableError(
                f'Could not select [{index}] {element.role} "{element.name}": {exc}'
            ) from exc

        # 3. Wait for DOM update
        try:  # noqa: SIM105
            await page.wait_for_timeout(500)
        except Exception:  # noqa: S110
            pass

        # 4. Re-extract page state
        rendered = await self._extract_and_render(page)

        logger.info(
            "browser_selected",
            index=index,
            role=element.role,
            name=element.name,
            value=value,
        )
        return self._page_state_output(rendered)
