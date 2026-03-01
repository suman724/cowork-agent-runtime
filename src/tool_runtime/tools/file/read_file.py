"""ReadFile tool — reads file contents with encoding detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    FileTooLargeError,
    ToolExecutionError,
    ToolPermissionError,
)
from tool_runtime.models import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    ExecutionContext,
    RawToolOutput,
)
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.platform.base import PlatformAdapter

_BINARY_CHECK_SIZE = 8192


class ReadFileTool(BaseTool):
    """Read the contents of a file."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    @property
    def name(self) -> str:
        return "ReadFile"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file at the given absolute path. "
            "Supports text files with automatic encoding detection. "
            "Use offset and limit for partial reads."
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
                    "description": "Absolute path to the file to read.",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from.",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return.",
                    "minimum": 1,
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
        offset: int | None = arguments.get("offset")
        limit: int | None = arguments.get("limit")

        validate_absolute_path(path)
        real_path = await self._platform.resolve_symlinks(path)
        validate_absolute_path(real_path)

        p = Path(real_path)
        if not p.exists():
            raise FileNotFoundToolError(f"File not found: {path}")

        if not p.is_file():
            raise ToolExecutionError(f"Path is not a file: {path}")

        try:
            file_size = p.stat().st_size
        except OSError as e:
            raise ToolPermissionError(f"Cannot access file: {e}") from e

        max_size = context.max_file_size_bytes or DEFAULT_MAX_FILE_SIZE_BYTES
        if file_size > max_size:
            raise FileTooLargeError(
                f"File is {file_size} bytes, exceeds limit of {max_size} bytes: {path}"
            )

        try:
            raw_bytes = p.read_bytes()
        except PermissionError as e:
            raise ToolPermissionError(f"Permission denied reading file: {e}") from e

        if _is_binary(raw_bytes):
            return RawToolOutput(output_text=f"Binary file, {file_size} bytes: {path}")

        content = _decode(raw_bytes, self._platform.default_encoding)
        content = self._platform.normalize_line_endings(content)

        if offset is not None or limit is not None:
            content = _apply_offset_limit(content, offset, limit)

        return RawToolOutput(output_text=content)


def _is_binary(data: bytes) -> bool:
    """Detect binary content by checking for null bytes in the first 8KB."""
    return b"\x00" in data[:_BINARY_CHECK_SIZE]


def _decode(data: bytes, platform_encoding: str) -> str:
    """Decode bytes with fallback chain: utf-8 → BOM → platform default → latin-1."""
    # Strip BOM if present
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8")

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass

    try:
        return data.decode(platform_encoding)
    except (UnicodeDecodeError, LookupError):
        pass

    return data.decode("latin-1")


def _apply_offset_limit(content: str, offset: int | None, limit: int | None) -> str:
    """Apply 1-based line offset and limit."""
    lines = content.splitlines(keepends=True)
    start = (offset - 1) if offset is not None else 0
    end = (start + limit) if limit is not None else len(lines)
    return "".join(lines[start:end])
