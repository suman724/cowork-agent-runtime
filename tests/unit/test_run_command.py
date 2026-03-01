"""Tests for RunCommand tool."""

from __future__ import annotations

import pytest

from tool_runtime.exceptions import ToolInputValidationError, ToolTimeoutError
from tool_runtime.models import ExecutionContext
from tool_runtime.platform.darwin import DarwinAdapter
from tool_runtime.tools.shell.run_command import RunCommandTool


@pytest.fixture
def platform() -> DarwinAdapter:
    return DarwinAdapter()


@pytest.fixture
def tool(platform: DarwinAdapter) -> RunCommandTool:
    return RunCommandTool(platform)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestRunCommand:
    async def test_echo_command(self, tool: RunCommandTool, context: ExecutionContext) -> None:
        result = await tool.execute({"command": "echo hello"}, context)
        assert "Exit code: 0" in result.output_text
        assert "hello" in result.output_text

    async def test_nonzero_exit_code(self, tool: RunCommandTool, context: ExecutionContext) -> None:
        result = await tool.execute({"command": "exit 42"}, context)
        assert "Exit code: 42" in result.output_text

    async def test_timeout_kills_process(
        self, tool: RunCommandTool, context: ExecutionContext
    ) -> None:
        with pytest.raises(ToolTimeoutError, match="timed out"):
            await tool.execute({"command": "sleep 60", "timeout_seconds": 1}, context)

    async def test_stdin_input(self, tool: RunCommandTool, context: ExecutionContext) -> None:
        result = await tool.execute({"command": "cat", "stdin": "hello from stdin"}, context)
        assert "hello from stdin" in result.output_text

    async def test_stderr_captured(self, tool: RunCommandTool, context: ExecutionContext) -> None:
        result = await tool.execute({"command": "echo error >&2"}, context)
        assert "stderr" in result.output_text
        assert "error" in result.output_text

    async def test_working_directory(self, tool: RunCommandTool, tmp_path: object) -> None:
        ctx = ExecutionContext(working_directory=str(tmp_path))
        result = await tool.execute({"command": "pwd"}, ctx)
        assert str(tmp_path) in result.output_text

    async def test_relative_working_directory_rejected(
        self, tool: RunCommandTool, context: ExecutionContext
    ) -> None:
        with pytest.raises(ToolInputValidationError, match="absolute"):
            await tool.execute({"command": "pwd", "working_directory": "relative/path"}, context)
