"""CreateDirectory tool — create directories without shell commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import (
    ToolExecutionError,
    ToolPermissionError,
)
from tool_runtime.models import RawToolOutput
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext
    from tool_runtime.platform.base import PlatformAdapter


class CreateDirectoryTool(BaseTool):
    """Create a directory at a given absolute path."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    @property
    def name(self) -> str:
        return "CreateDirectory"

    @property
    def description(self) -> str:
        return (
            "Create a directory at the given absolute path. "
            "By default creates parent directories as needed. "
            "Returns a confirmation message if the directory already exists."
        )

    @property
    def capability(self) -> str:
        return "File.Write"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the directory to create.",
                },
                "create_parents": {
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
        context: ExecutionContext,  # noqa: ARG002
    ) -> RawToolOutput:
        self.validate_input(arguments)
        path: str = arguments["path"]
        create_parents: bool = arguments.get("create_parents", True)

        validate_absolute_path(path)

        # Resolve symlinks on parent directory (target may not exist yet)
        p = Path(path)
        if p.parent.exists():
            real_parent = await self._platform.resolve_symlinks(str(p.parent))
            target = Path(real_parent) / p.name
        else:
            target = p

        if target.exists():
            if target.is_dir():
                return RawToolOutput(output_text=f"Directory already exists: {path}")
            raise ToolExecutionError(f"Path exists but is a file: {path}")

        try:
            target.mkdir(parents=create_parents, exist_ok=False)
        except FileNotFoundError as e:
            raise ToolExecutionError(
                f"Parent directory does not exist (set create_parents=true): {e}"
            ) from e
        except PermissionError as e:
            raise ToolPermissionError(f"Permission denied creating directory: {e}") from e
        except OSError as e:
            raise ToolExecutionError(f"Failed to create directory: {e}") from e

        return RawToolOutput(output_text=f"Created directory: {path}")
