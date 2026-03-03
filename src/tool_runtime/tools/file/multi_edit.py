"""MultiEdit tool — batch multiple edits to one file atomically."""

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
from tool_runtime.validation import validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput
    from tool_runtime.platform.base import PlatformAdapter


class MultiEditTool(BaseTool):
    """Apply multiple find-and-replace edits to a file atomically."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    @property
    def name(self) -> str:
        return "MultiEdit"

    @property
    def description(self) -> str:
        return (
            "Apply multiple find-and-replace edits to a single file atomically. "
            "Each edit's old_text must appear exactly once. Edits are applied in "
            "order. All changes are written at once — if any edit fails validation, "
            "the file is not modified. Returns a unified diff of all changes."
        )

    @property
    def capability(self) -> str:
        return "File.Write"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path", "edits"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit.",
                },
                "edits": {
                    "type": "array",
                    "description": "Ordered list of find-and-replace edits to apply.",
                    "items": {
                        "type": "object",
                        "required": ["old_text", "new_text"],
                        "properties": {
                            "old_text": {
                                "type": "string",
                                "description": "Exact text to find. Must appear exactly once.",
                            },
                            "new_text": {
                                "type": "string",
                                "description": "Replacement text.",
                            },
                        },
                        "additionalProperties": False,
                    },
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
        edits: list[dict[str, str]] = arguments["edits"]

        if not edits:
            raise ToolInputValidationError("edits array cannot be empty")

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

        original_content = content

        # Filter out no-op edits (old_text == new_text)
        effective_edits = [
            (i, edit) for i, edit in enumerate(edits) if edit["old_text"] != edit["new_text"]
        ]

        if not effective_edits:
            from tool_runtime.models import RawToolOutput

            return RawToolOutput(output_text="No changes needed")

        # Validation pass: verify all old_texts before mutating
        for i, edit in effective_edits:
            old_text = edit["old_text"]
            count = content.count(old_text)
            if count == 0:
                raise ToolExecutionError(f"edits[{i}].old_text not found in file")
            if count > 1:
                raise ToolExecutionError(
                    f"edits[{i}].old_text matches {count} times, must be unique"
                )

        # Check for overlapping edits: verify no two old_texts overlap in position
        _check_overlapping_edits(content, effective_edits)

        # Apply edits sequentially
        for i, edit in effective_edits:
            old_text = edit["old_text"]
            new_text = edit["new_text"]

            # Re-validate that this edit still has exactly 1 match after prior edits
            count = content.count(old_text)
            if count == 0:
                raise ToolExecutionError(
                    f"edits[{i}].old_text not found after applying previous edits"
                )
            if count > 1:
                raise ToolExecutionError(
                    f"edits[{i}].old_text matches {count} times after applying previous edits, "
                    "must be unique"
                )

            content = content.replace(old_text, new_text, 1)

        # Atomic write (same pattern as EditFile)
        parent_dir = p.parent
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(parent_dir), prefix=".multiedit_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                Path(tmp_path).replace(real_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()
                raise
        except PermissionError as e:
            raise ToolPermissionError(f"Permission denied writing file: {e}") from e
        except OSError as e:
            raise ToolExecutionError(f"Failed to write file: {e}") from e

        # Generate unified diff (original → final)
        diff_text = _generate_diff(original_content, content, path)
        output_text = f"Edited file: {path} ({len(effective_edits)} edits)\n\n{diff_text}"

        max_output = context.max_output_bytes
        return maybe_extract_artifact(output_text, "file_diff", f"{p.name}.diff", max_output)


def _check_overlapping_edits(
    content: str,
    edits: list[tuple[int, dict[str, str]]],
) -> None:
    """Verify no two edits target overlapping text ranges in the original content."""
    # Build list of (start, end, edit_index) for each edit's match position
    ranges: list[tuple[int, int, int]] = []
    for i, edit in edits:
        old_text = edit["old_text"]
        start = content.find(old_text)
        if start == -1:
            continue  # Already handled in validation pass
        end = start + len(old_text)
        ranges.append((start, end, i))

    # Sort by start position and check for overlaps
    ranges.sort()
    for idx in range(len(ranges) - 1):
        _, end_a, edit_idx_a = ranges[idx]
        start_b, _, edit_idx_b = ranges[idx + 1]
        if end_a > start_b:
            raise ToolExecutionError(
                f"edits[{edit_idx_a}] and edits[{edit_idx_b}] target overlapping text ranges"
            )


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
