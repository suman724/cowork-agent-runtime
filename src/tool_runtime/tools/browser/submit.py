"""BrowserSubmit — submit a form with mandatory approval checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from tool_runtime.exceptions import (
    BrowserSubmitApprovalRequiredError,
)
from tool_runtime.tools.browser.base_browser_tool import BaseBrowserTool
from tool_runtime.tools.browser.sensitive_detector import redact_form_value

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

logger = structlog.get_logger(__name__)


class BrowserSubmitTool(BaseBrowserTool):
    """Submit a form — always requires user approval."""

    @property
    def name(self) -> str:
        return "BrowserSubmit"

    @property
    def description(self) -> str:
        return (
            "Submit a form by clicking the submit button. Always requires user approval. "
            "Provide a description of what is being submitted."
        )

    @property
    def capability(self) -> str:
        return "Browser.Submit"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Index of the submit button or form element.",
                },
                "description": {
                    "type": "string",
                    "description": "What is being submitted (shown in approval dialog).",
                },
            },
            "required": ["index", "description"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ExecutionContext,  # noqa: ARG002
    ) -> RawToolOutput:
        self.validate_input(arguments)
        index: int = arguments["index"]
        description: str = arguments["description"]

        page = await self._get_page()

        # 1. Resolve element
        element, _snapshot = await self._resolve_element(page, index)

        # 2. Check if this is a retry after approval
        if not arguments.get("_approved"):
            # First call — extract form data and trigger approval
            form_data = await self._extract_form_data(page)
            screenshot_b64 = await self._capture_screenshot_base64(page)

            raise BrowserSubmitApprovalRequiredError(
                description=description,
                url=page.url,
                form_data=form_data,
                screenshot_base64=screenshot_b64,
            )

        # 3. Approved — click the submit element
        try:
            locator = page.get_by_role(
                element.role,  # type: ignore[arg-type]
                name=element.name,
            )
            await locator.first.click()
        except Exception as exc:
            from tool_runtime.exceptions import BrowserElementNotInteractableError

            raise BrowserElementNotInteractableError(
                f'Could not submit [{index}] {element.role} "{element.name}": {exc}'
            ) from exc

        # 4. Wait for navigation/response
        try:  # noqa: SIM105
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:  # noqa: S110
            pass

        # 5. Re-extract page state
        rendered = await self._extract_and_render(page)

        logger.info("browser_submitted", index=index, description=description)
        return self._page_state_output(rendered)

    async def _extract_form_data(self, page: Any) -> list[dict[str, str]]:
        """Extract visible form field labels and redacted values."""
        try:
            fields = await page.evaluate("""() => {
                const inputs = document.querySelectorAll(
                    'input, select, textarea'
                );
                return Array.from(inputs).map(el => ({
                    type: el.type || el.tagName.toLowerCase(),
                    name: el.name || el.id || '',
                    autocomplete: el.autocomplete || '',
                    value: el.value || '',
                    label: (
                        el.labels?.[0]?.textContent?.trim() ||
                        el.getAttribute('aria-label') ||
                        el.placeholder ||
                        el.name ||
                        el.id ||
                        ''
                    ),
                }));
            }""")
        except Exception:
            logger.warning("form_data_extraction_failed", exc_info=True)
            return []

        result: list[dict[str, str]] = []
        for field in fields:
            label = field.get("label", "")
            value = field.get("value", "")
            if not label and not value:
                continue

            redacted_value = redact_form_value(
                field_type=field.get("type", ""),
                autocomplete=field.get("autocomplete", ""),
                field_name=field.get("name", ""),
                value=value,
            )
            result.append({"label": label, "value": redacted_value})

        return result
