"""RunCommand tool — execute shell commands with timeout and process group kill."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import ToolExecutionError, ToolTimeoutError
from tool_runtime.models import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ExecutionContext,
    RawToolOutput,
)
from tool_runtime.output.artifacts import maybe_extract_artifact
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import validate_absolute_path

if TYPE_CHECKING:
    from tool_runtime.platform.base import PlatformAdapter


class RunCommandTool(BaseTool):
    """Execute a shell command."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    @property
    def name(self) -> str:
        return "RunCommand"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its stdout, stderr, and exit code. "
            "Commands are run via the system shell with a configurable timeout."
        )

    @property
    def capability(self) -> str:
        return "Shell.Exec"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["command", "description"],
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "A brief explanation of what this command does "
                        "(e.g., 'Install project dependencies')."
                    ),
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds.",
                    "minimum": 1,
                    "default": DEFAULT_COMMAND_TIMEOUT_SECONDS,
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory for the command.",
                },
                "stdin": {
                    "type": "string",
                    "description": "Input to pass to the command's stdin.",
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
        command: str = arguments["command"]
        description: str = arguments["description"]
        timeout: int = arguments.get("timeout_seconds", DEFAULT_COMMAND_TIMEOUT_SECONDS)
        cwd: str | None = arguments.get("working_directory") or context.working_directory
        stdin_data: str | None = arguments.get("stdin")

        if cwd is not None:
            validate_absolute_path(cwd)

        if context.command_timeout_seconds is not None:
            timeout = min(timeout, context.command_timeout_seconds)

        # Use process group for reliable tree kill on Unix
        preexec = os.setsid if sys.platform != "win32" else None

        try:
            process = await asyncio.create_subprocess_exec(
                self._platform.shell_executable,
                self._platform.shell_flag,
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
                cwd=cwd,
                preexec_fn=preexec,
            )
        except FileNotFoundError as e:
            raise ToolExecutionError(f"Shell not found: {e}") from e
        except OSError as e:
            raise ToolExecutionError(f"Failed to start process: {e}") from e

        try:
            stdin_bytes = stdin_data.encode("utf-8") if stdin_data else None
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=stdin_bytes),
                timeout=timeout,
            )
        except TimeoutError as e:
            await self._platform.kill_process_tree(process.pid)
            raise ToolTimeoutError(f"Command timed out after {timeout}s: {command}") from e

        stdout = self._platform.normalize_line_endings(
            stdout_bytes.decode("utf-8", errors="replace")
        )
        stderr = self._platform.normalize_line_endings(
            stderr_bytes.decode("utf-8", errors="replace")
        )

        exit_code = process.returncode if process.returncode is not None else 0
        output_text = f"# {description}\n{_format_output(exit_code, stdout, stderr)}"

        max_output = context.max_output_bytes
        result = maybe_extract_artifact(
            output_text, "command_output", "command_output.txt", max_output
        )

        return result


def _format_output(exit_code: int, stdout: str, stderr: str) -> str:
    """Format command output with exit code, stdout, and stderr."""
    parts = [f"Exit code: {exit_code}"]
    if stdout:
        parts.append(f"--- stdout ---\n{stdout}")
    if stderr:
        parts.append(f"--- stderr ---\n{stderr}")
    return "\n".join(parts)
