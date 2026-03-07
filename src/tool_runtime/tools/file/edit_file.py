"""EditFile tool — exact-match find-and-replace editing."""

from __future__ import annotations

import contextlib
import difflib
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    ToolExecutionError,
    ToolInputValidationError,
    ToolPermissionError,
)
from tool_runtime.output.artifacts import maybe_extract_artifact
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import resolve_relative_path, validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput
    from tool_runtime.platform.base import PlatformAdapter


class EditFileTool(BaseTool):
    """Edit a file by replacing an exact text match."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    @property
    def name(self) -> str:
        return "EditFile"

    @property
    def description(self) -> str:
        return (
            "Edit a file by finding and replacing an exact text match. "
            "The old_text must appear exactly once in the file. "
            "Safer than WriteFile for targeted edits — only changes "
            "the matched section. Returns a unified diff of the changes."
        )

    @property
    def capability(self) -> str:
        return "File.Write"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path", "old_text", "new_text"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit.",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace. Must appear exactly once.",
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text.",
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
        old_text: str = arguments["old_text"]
        new_text: str = arguments["new_text"]

        if old_text == new_text:
            raise ToolInputValidationError("old_text and new_text are identical — no change needed")

        path = resolve_relative_path(path, context.working_directory)
        validate_absolute_path(path)
        real_path = await self._platform.resolve_symlinks(path)
        validate_absolute_path(real_path)

        p = Path(real_path)
        if not p.exists():
            raise FileNotFoundToolError(f"File not found: {path}")

        if not p.is_file():
            raise ToolExecutionError(f"Path is not a file: {path}")

        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except PermissionError as e:
            raise ToolPermissionError(f"Permission denied reading file: {e}") from e

        # Verify old_text appears exactly once
        count = content.count(old_text)
        if count == 0:
            raise ToolExecutionError("old_text not found in file")
        if count > 1:
            raise ToolExecutionError(f"old_text matches {count} times in file, must be unique")

        new_content = content.replace(old_text, new_text, 1)

        # Atomic write (same pattern as WriteFile)
        parent_dir = p.parent
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(parent_dir), prefix=".edit_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(new_content)
                Path(tmp_path).replace(real_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()
                raise
        except PermissionError as e:
            raise ToolPermissionError(f"Permission denied writing file: {e}") from e
        except OSError as e:
            raise ToolExecutionError(f"Failed to write file: {e}") from e

        # Generate unified diff
        diff_text = _generate_diff(content, new_content, path)
        output_text = f"Edited file: {path}\n\n{diff_text}"

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
