"""FindFiles tool — glob-pattern file discovery across a directory tree."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    ToolExecutionError,
)
from tool_runtime.output.artifacts import maybe_extract_artifact
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

_DEFAULT_MAX_RESULTS = 100
_ABSOLUTE_MAX_RESULTS = 500


class FindFilesTool(BaseTool):
    """Find files matching a glob pattern in a directory tree."""

    @property
    def name(self) -> str:
        return "FindFiles"

    @property
    def description(self) -> str:
        return (
            "Find files matching a glob pattern recursively in a directory. "
            "Returns relative paths from the search directory. "
            "Use patterns like '*.py', '**/*.json', 'test_*.py'."
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
                    "description": "Glob pattern to match (e.g., '*.py', '**/*.json').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "minimum": 1,
                    "maximum": _ABSOLUTE_MAX_RESULTS,
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
        pattern: str = arguments["pattern"]
        max_results: int = min(
            arguments.get("max_results", _DEFAULT_MAX_RESULTS),
            _ABSOLUTE_MAX_RESULTS,
        )

        validate_absolute_path(directory)

        p = Path(directory)
        if not p.exists():
            raise FileNotFoundToolError(f"Directory not found: {directory}")

        if not p.is_dir():
            raise ToolExecutionError(f"Path is not a directory: {directory}")

        matches: list[str] = []
        truncated = False
        for match in p.rglob(pattern):
            if len(matches) >= max_results:
                truncated = True
                break
            try:
                rel = match.relative_to(p)
                matches.append(str(rel))
            except ValueError:
                continue

        matches.sort()

        if not matches:
            output_text = f"No files matching '{pattern}' found in {directory}"
        else:
            output_text = "\n".join(matches)
            if truncated:
                output_text += f"\n\n[Results truncated at {max_results} matches]"

        max_output = context.max_output_bytes
        return maybe_extract_artifact(output_text, "file_list", "find_results.txt", max_output)
