"""ViewImage tool — read an image file and return base64 for multimodal LLM."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    FileTooLargeError,
    ToolExecutionError,
    ToolInputValidationError,
    ToolPermissionError,
)
from tool_runtime.models import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    ImageContent,
    RawToolOutput,
)
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import resolve_relative_path, validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext
    from tool_runtime.platform.base import PlatformAdapter

_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _human_size(size: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size_f = size / 1024
        if unit != "MB":
            size = int(size_f)
    return f"{size_f:.1f}GB"


class ViewImageTool(BaseTool):
    """Read an image file and return it as base64 for multimodal LLM."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    @property
    def name(self) -> str:
        return "ViewImage"

    @property
    def description(self) -> str:
        return (
            "Read an image file and return it for visual analysis. "
            "Supports PNG, JPEG, GIF, WebP, and BMP formats. "
            "The image will be included in the conversation for the LLM to see."
        )

    @property
    def capability(self) -> str:
        return "File.Read"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the image file.",
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> RawToolOutput:
        self.validate_input(arguments)
        path: str = arguments["path"]

        path = resolve_relative_path(path, context.working_directory)
        validate_absolute_path(path)
        real_path = await self._platform.resolve_symlinks(path)
        validate_absolute_path(real_path)

        p = Path(real_path)
        if not p.exists():
            raise FileNotFoundToolError(f"Image file not found: {path}")

        if not p.is_file():
            raise ToolExecutionError(f"Path is not a file: {path}")

        ext = p.suffix.lower()
        media_type = _SUPPORTED_EXTENSIONS.get(ext)
        if media_type is None:
            supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS.keys()))
            raise ToolInputValidationError(
                f"Unsupported image format '{ext}'. Supported: {supported}"
            )

        try:
            file_size = p.stat().st_size
        except OSError as e:
            raise ToolPermissionError(f"Cannot access file: {e}") from e

        max_size = context.max_file_size_bytes or DEFAULT_MAX_FILE_SIZE_BYTES
        if file_size > max_size:
            raise FileTooLargeError(
                f"Image is {file_size} bytes, exceeds limit of {max_size} bytes: {path}"
            )

        try:
            data = p.read_bytes()
        except PermissionError as e:
            raise ToolPermissionError(f"Permission denied reading image: {e}") from e

        b64 = base64.b64encode(data).decode("ascii")

        return RawToolOutput(
            output_text=f"Image: {path} ({_human_size(file_size)})",
            image_content=ImageContent(media_type=media_type, base64_data=b64),
        )
