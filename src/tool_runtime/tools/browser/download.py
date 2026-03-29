"""BrowserDownload — download files from the browser to workspace."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from tool_runtime.exceptions import (
    BrowserDownloadError,
    BrowserDownloadTimeoutError,
)
from tool_runtime.models import RawToolOutput
from tool_runtime.tools.browser.base_browser_tool import BaseBrowserTool
from tool_runtime.validation import validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext

logger = structlog.get_logger(__name__)

_DOWNLOAD_TIMEOUT_MS = 120_000  # 2 minutes


class BrowserDownloadTool(BaseBrowserTool):
    """Download a file from the browser to the workspace."""

    @property
    def name(self) -> str:
        return "BrowserDownload"

    @property
    def description(self) -> str:
        return (
            "Download a file by clicking a download link/button or navigating to a URL. "
            "Always requires approval. Downloaded file is saved to the workspace."
        )

    @property
    def capability(self) -> str:
        return "Browser.Download"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Index of the download link/button.",
                },
                "url": {
                    "type": "string",
                    "description": "Direct URL to download (mutually exclusive with index).",
                },
                "savePath": {
                    "type": "string",
                    "description": "Absolute path where the file should be saved.",
                },
            },
            "required": ["savePath"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> RawToolOutput:
        self.validate_input(arguments)
        index: int | None = arguments.get("index")
        url: str | None = arguments.get("url")
        save_path: str = arguments["savePath"]

        if not index and not url:
            raise BrowserDownloadError("Either 'index' or 'url' is required")

        # 1. Validate save path is absolute
        validate_absolute_path(save_path)

        page = await self._get_page()

        # 2. Start download
        try:
            async with page.expect_download(timeout=_DOWNLOAD_TIMEOUT_MS) as download_info:
                if index is not None:
                    element, _snapshot = await self._resolve_element(page, index)
                    locator = page.get_by_role(
                        element.role,  # type: ignore[arg-type]
                        name=element.name,
                    )
                    await locator.first.click()
                elif url:
                    await page.goto(url)

            download = await download_info.value

        except TimeoutError as exc:
            raise BrowserDownloadTimeoutError(
                f"Download timed out after {_DOWNLOAD_TIMEOUT_MS // 1000}s"
            ) from exc
        except Exception as exc:
            raise BrowserDownloadError(f"Download failed: {exc}") from exc

        # 3. Check file size
        if context.max_file_size_bytes:
            path_obj = Path(await download.path() or "")
            if path_obj.exists() and path_obj.stat().st_size > context.max_file_size_bytes:
                await download.delete()
                raise BrowserDownloadError(
                    f"Downloaded file exceeds max size ({context.max_file_size_bytes} bytes)"
                )

        # 4. Save to workspace
        try:
            await download.save_as(save_path)
        except Exception as exc:
            raise BrowserDownloadError(f"Failed to save download: {exc}") from exc

        filename = download.suggested_filename
        logger.info("browser_downloaded", filename=filename, save_path=save_path)

        return RawToolOutput(output_text=f"Downloaded '{filename}' to {save_path}")
