"""ExecuteCode tool — run Python scripts with rich output support."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from tool_runtime.code.executor import PythonExecutor
from tool_runtime.exceptions import ToolExecutionError
from tool_runtime.models import ImageContent, RawToolOutput
from tool_runtime.output.artifacts import maybe_extract_artifact
from tool_runtime.tools.base import BaseTool

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext
    from tool_runtime.platform.base import PlatformAdapter

DEFAULT_CODE_TIMEOUT_SECONDS = 120
MAX_CODE_TIMEOUT_SECONDS = 600


class ExecuteCodeTool(BaseTool):
    """Execute a Python script and return stdout, stderr, exit code, and images."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._executor = PythonExecutor(platform)

    @property
    def name(self) -> str:
        return "ExecuteCode"

    @property
    def description(self) -> str:
        return (
            "Execute a Python script and return its stdout, stderr, and exit code. "
            "Each execution is independent — write complete, self-contained scripts. "
            "Matplotlib plots are captured automatically (plt.show() saves images). "
            "Use for: calculations, data analysis, testing code, prototyping."
        )

    @property
    def capability(self) -> str:
        return "Code.Execute"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["code", "description"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "A brief explanation of what this code does "
                        "(e.g., 'Calculate first 20 Fibonacci numbers')."
                    ),
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds.",
                    "minimum": 1,
                    "maximum": MAX_CODE_TIMEOUT_SECONDS,
                    "default": DEFAULT_CODE_TIMEOUT_SECONDS,
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
        code: str = arguments["code"]
        description: str = arguments["description"]
        timeout: int = arguments.get("timeout_seconds", DEFAULT_CODE_TIMEOUT_SECONDS)

        # Policy timeout takes precedence
        if context.max_execution_time_seconds is not None:
            timeout = min(timeout, context.max_execution_time_seconds)

        try:
            result = await self._executor.execute(
                code,
                working_directory=context.working_directory,
                timeout=timeout,
            )
        except OSError as e:
            raise ToolExecutionError(f"Failed to execute code: {e}") from e

        # Format output (same structure as RunCommand for LLM familiarity)
        output_text = f"# {description}\n{_format_code_output(result.exit_code, result.stdout, result.stderr, result.execution_time, result.timed_out)}"

        # Extract artifact if output is large
        max_output = context.max_output_bytes
        raw = maybe_extract_artifact(output_text, "code_output", "code_output.txt", max_output)

        # Attach first image as ImageContent (same pattern as ViewImage)
        if result.images:
            b64 = base64.b64encode(result.images[0]).decode("ascii")
            raw.image_content = ImageContent(media_type="image/png", base64_data=b64)

        return raw


def _format_code_output(
    exit_code: int,
    stdout: str,
    stderr: str,
    execution_time: float,
    timed_out: bool,
) -> str:
    """Format code execution output."""
    parts = [f"Exit code: {exit_code}"]
    if stdout:
        parts.append(f"--- stdout ---\n{stdout}")
    if stderr:
        parts.append(f"--- stderr ---\n{stderr}")
    parts.append(f"[Execution time: {execution_time:.2f}s]")
    if timed_out:
        parts.append("[TIMED OUT]")
    return "\n".join(parts)
