"""DeleteFile tool — delete files (not directories)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    ToolExecutionError,
    ToolPermissionError,
)
from tool_runtime.models import ExecutionContext, RawToolOutput
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import resolve_relative_path, validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.platform.base import PlatformAdapter


class DeleteFileTool(BaseTool):
    """Delete a file."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    @property
    def name(self) -> str:
        return "DeleteFile"

    @property
    def description(self) -> str:
        return "Delete a file at the given absolute path. Cannot delete directories."

    @property
    def capability(self) -> str:
        return "File.Delete"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to delete.",
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
            raise FileNotFoundToolError(f"File not found: {path}")

        if p.is_dir():
            raise ToolExecutionError(f"Cannot delete a directory: {path}")

        try:
            p.unlink()
        except PermissionError as e:
            raise ToolPermissionError(f"Permission denied deleting file: {e}") from e
        except OSError as e:
            raise ToolExecutionError(f"Failed to delete file: {e}") from e

        return RawToolOutput(output_text=f"Deleted file: {path}")
