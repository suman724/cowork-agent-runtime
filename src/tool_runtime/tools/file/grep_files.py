"""GrepFiles tool — regex search across files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    ToolExecutionError,
    ToolInputValidationError,
)
from tool_runtime.output.artifacts import maybe_extract_artifact
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

_DEFAULT_MAX_RESULTS = 50
_ABSOLUTE_MAX_RESULTS = 200
_MAX_CONTEXT_LINES = 5
_BINARY_CHECK_SIZE = 512


def _is_binary(data: bytes) -> bool:
    """Detect binary content by checking for null bytes in the first 512 bytes."""
    return b"\x00" in data[:_BINARY_CHECK_SIZE]


class GrepFilesTool(BaseTool):
    """Search file contents with regex patterns."""

    @property
    def name(self) -> str:
        return "GrepFiles"

    @property
    def description(self) -> str:
        return (
            "Search for a regex pattern across files in a directory. "
            "Returns matching lines with file paths and line numbers. "
            "Optionally filter by file glob and include context lines."
        )

    @property
    def capability(self) -> str:
        return "File.Read"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["directory", "pattern"],
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Absolute path to the directory to search in.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for.",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py'). Defaults to '*'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return.",
                    "minimum": 1,
                    "maximum": _ABSOLUTE_MAX_RESULTS,
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines before and after each match.",
                    "minimum": 0,
                    "maximum": _MAX_CONTEXT_LINES,
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
        directory: str = arguments["directory"]
        pattern_str: str = arguments["pattern"]
        file_glob: str = arguments.get("file_glob", "*")
        max_results: int = min(
            arguments.get("max_results", _DEFAULT_MAX_RESULTS),
            _ABSOLUTE_MAX_RESULTS,
        )
        context_lines: int = min(
            arguments.get("context_lines", 0),
            _MAX_CONTEXT_LINES,
        )

        validate_absolute_path(directory)

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundToolError(f"Directory not found: {directory}")

        if not dir_path.is_dir():
            raise ToolExecutionError(f"Path is not a directory: {directory}")

        try:
            regex = re.compile(pattern_str)
        except re.error as e:
            raise ToolInputValidationError(f"Invalid regex pattern: {e}") from e

        results: list[str] = []
        truncated = False

        for file_path in sorted(dir_path.rglob(file_glob)):
            if truncated:
                break
            if not file_path.is_file():
                continue

            try:
                raw = file_path.read_bytes()
            except (PermissionError, OSError):
                continue

            if _is_binary(raw):
                continue

            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue

            lines = content.splitlines()
            rel_path = file_path.relative_to(dir_path)

            for i, line in enumerate(lines):
                if regex.search(line):
                    if context_lines > 0:
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        for j in range(start, end):
                            prefix = ">" if j == i else " "
                            results.append(f"{rel_path}:{j + 1}:{prefix} {lines[j]}")
                        results.append("")  # separator between groups
                    else:
                        results.append(f"{rel_path}:{i + 1}: {line}")

                    if len(results) >= max_results:
                        truncated = True
                        break

        if not results:
            output_text = f"No matches for pattern '{pattern_str}' in {directory}"
        else:
            output_text = "\n".join(results)
            if truncated:
                output_text += f"\n\n[Results truncated at {max_results} matches]"

        max_output = context.max_output_bytes
        return maybe_extract_artifact(output_text, "grep_results", "grep_results.txt", max_output)
