"""MoveFile tool — move or rename files and directories."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    ToolExecutionError,
    ToolPermissionError,
)
from tool_runtime.models import RawToolOutput
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import resolve_relative_path, validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext
    from tool_runtime.platform.base import PlatformAdapter


class MoveFileTool(BaseTool):
    """Move or rename a file or directory."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    @property
    def name(self) -> str:
        return "MoveFile"

    @property
    def description(self) -> str:
        return (
            "Move or rename a file or directory. "
            "Creates destination parent directories if needed. "
            "Set overwrite=true to replace an existing destination."
        )

    @property
    def capability(self) -> str:
        return "File.Write"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["source", "destination"],
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Absolute path to the file or directory to move.",
                },
                "destination": {
                    "type": "string",
                    "description": "Absolute path to move to.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Replace destination if it already exists.",
                    "default": False,
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
        source: str = arguments["source"]
        destination: str = arguments["destination"]
        overwrite: bool = arguments.get("overwrite", False)

        source = resolve_relative_path(source, context.working_directory)
        destination = resolve_relative_path(destination, context.working_directory)
        validate_absolute_path(source)
        validate_absolute_path(destination)

        # Resolve symlinks
        real_source = await self._platform.resolve_symlinks(source)
        validate_absolute_path(real_source)

        src = Path(real_source)
        if not src.exists():
            raise FileNotFoundToolError(f"Source not found: {source}")

        # Resolve destination parent symlinks (dest may not exist yet)
        dest = Path(destination)
        if dest.parent.exists():
            real_parent = await self._platform.resolve_symlinks(str(dest.parent))
            real_dest = Path(real_parent) / dest.name
        else:
            real_dest = dest

        if real_dest.exists() and not overwrite:
            raise ToolExecutionError(
                f"Destination already exists: {destination}. Set overwrite=true to replace."
            )

        # Create destination parent directories if needed
        try:
            real_dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ToolExecutionError(f"Failed to create destination directories: {e}") from e

        try:
            shutil.move(str(src), str(real_dest))
        except PermissionError as e:
            raise ToolPermissionError(f"Permission denied moving file: {e}") from e
        except OSError as e:
            raise ToolExecutionError(f"Failed to move file: {e}") from e

        return RawToolOutput(output_text=f"Moved: {source} \u2192 {destination}")
