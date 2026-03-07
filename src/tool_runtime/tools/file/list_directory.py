"""ListDirectory tool — list files and directories at a path."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    ToolExecutionError,
    ToolPermissionError,
)
from tool_runtime.output.artifacts import maybe_extract_artifact
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import resolve_relative_path, validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput
    from tool_runtime.platform.base import PlatformAdapter


def _human_size(size: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size_f = size / 1024
        size = int(size_f) if unit == "B" else size
        if unit != "GB":
            size = int(size_f)
    return f"{size_f:.1f}TB"


class ListDirectoryTool(BaseTool):
    """List files and directories at a given path."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    @property
    def name(self) -> str:
        return "ListDirectory"

    @property
    def description(self) -> str:
        return (
            "List files and directories at the given absolute path. "
            "Returns directory entries with type indicators (d for directory, "
            "f for file) and file sizes."
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
                    "description": "Absolute path to the directory to list.",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (dot-prefixed). Defaults to false.",
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
        include_hidden: bool = arguments.get("include_hidden", False)

        path = resolve_relative_path(path, context.working_directory)
        validate_absolute_path(path)
        real_path = await self._platform.resolve_symlinks(path)
        validate_absolute_path(real_path)

        p = Path(real_path)
        if not p.exists():
            raise FileNotFoundToolError(f"Directory not found: {path}")

        if not p.is_dir():
            raise ToolExecutionError(f"Path is not a directory: {path}")

        try:
            entries = sorted(p.iterdir(), key=lambda e: e.name)
        except PermissionError as e:
            raise ToolPermissionError(f"Permission denied listing directory: {e}") from e

        lines: list[str] = []
        for entry in entries:
            if not include_hidden and entry.name.startswith("."):
                continue
            if entry.is_dir():
                lines.append(f"d {entry.name}")
            else:
                try:
                    size = entry.stat().st_size
                    lines.append(f"f {entry.name} ({_human_size(size)})")
                except OSError:
                    lines.append(f"f {entry.name}")

        output_text = f"Directory is empty: {path}" if not lines else "\n".join(lines)

        max_output = context.max_output_bytes
        return maybe_extract_artifact(output_text, "directory_listing", "listing.txt", max_output)
