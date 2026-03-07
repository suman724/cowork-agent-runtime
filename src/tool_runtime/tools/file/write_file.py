"""WriteFile tool — atomic file write with diff generation."""

from __future__ import annotations

import contextlib
import difflib
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import (
    ToolExecutionError,
    ToolPermissionError,
)
from tool_runtime.output.artifacts import maybe_extract_artifact
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import resolve_relative_path, validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput
    from tool_runtime.platform.base import PlatformAdapter


class WriteFileTool(BaseTool):
    """Write content to a file atomically."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    @property
    def name(self) -> str:
        return "WriteFile"

    @property
    def description(self) -> str:
        return (
            "Write content to a file at the given absolute path. "
            "Creates parent directories if needed. Uses atomic write "
            "(write to temp file, then rename) to prevent corruption. "
            "Returns a unified diff of the changes."
        )

    @property
    def capability(self) -> str:
        return "File.Write"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path", "content"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to write to.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
                "create_directories": {
                    "type": "boolean",
                    "description": "Create parent directories if they don't exist.",
                    "default": True,
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
        content: str = arguments["content"]
        create_dirs: bool = arguments.get("create_directories", True)

        path = resolve_relative_path(path, context.working_directory)
        validate_absolute_path(path)
        p = Path(path)

        # Resolve symlinks: full path if file exists, parent-only for new files
        if p.exists():
            target_path = await self._platform.resolve_symlinks(path)
        else:
            real_parent = await self._platform.resolve_symlinks(str(p.parent))
            target_path = str(Path(real_parent) / p.name)
        validate_absolute_path(target_path)

        # Read existing content for diff
        old_content = ""
        is_new = True
        target = Path(target_path)
        if target.exists():
            is_new = False
            try:
                old_content = target.read_text(encoding="utf-8", errors="replace")
            except PermissionError as e:
                raise ToolPermissionError(f"Permission denied reading existing file: {e}") from e

        # Create parent directories if needed
        parent_dir = target.parent
        if create_dirs and not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ToolExecutionError(f"Failed to create directories: {e}") from e

        # Atomic write: write to temp file in same directory, then rename
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(parent_dir), prefix=".write_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                Path(tmp_path).replace(target_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()
                raise
        except PermissionError as e:
            raise ToolPermissionError(f"Permission denied writing file: {e}") from e
        except OSError as e:
            raise ToolExecutionError(f"Failed to write file: {e}") from e

        # Generate diff
        diff_text = _generate_diff(old_content, content, path)
        action = "Created" if is_new else "Updated"
        output_text = f"{action} file: {path}\n\n{diff_text}"

        max_output = context.max_output_bytes
        return maybe_extract_artifact(output_text, "file_diff", f"{p.name}.diff", max_output)


def _generate_diff(old_content: str, new_content: str, path: str) -> str:
    """Generate a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    name = Path(path).name

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{name}",
        tofile=f"b/{name}",
    )
    return "".join(diff)
