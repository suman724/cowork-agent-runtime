"""Tests for ExecuteCodeTool — input validation, output formatting, image handling."""

from __future__ import annotations

import pytest

from tool_runtime.exceptions import ToolInputValidationError
from tool_runtime.models import ExecutionContext
from tool_runtime.platform.detection import get_platform
from tool_runtime.tools.code.execute_code import ExecuteCodeTool


@pytest.fixture
def tool() -> ExecuteCodeTool:
    return ExecuteCodeTool(get_platform())


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestExecuteCodeTool:
    async def test_simple_execution(self, tool: ExecuteCodeTool, context: ExecutionContext) -> None:
        result = await tool.execute(
            {"code": "print(2 + 2)", "description": "Add two numbers"}, context
        )
        assert "Exit code: 0" in result.output_text
        assert "4" in result.output_text
        assert "Add two numbers" in result.output_text

    async def test_execution_time_in_output(
        self, tool: ExecuteCodeTool, context: ExecutionContext
    ) -> None:
        result = await tool.execute(
            {"code": "print('fast')", "description": "Fast script"}, context
        )
        assert "Execution time:" in result.output_text

    async def test_error_output(self, tool: ExecuteCodeTool, context: ExecutionContext) -> None:
        result = await tool.execute(
            {"code": "raise ValueError('boom')", "description": "Error test"}, context
        )
        assert "Exit code: 1" in result.output_text
        assert "ValueError" in result.output_text
        assert "boom" in result.output_text

    async def test_timeout_output(self, tool: ExecuteCodeTool, context: ExecutionContext) -> None:
        result = await tool.execute(
            {
                "code": "import time; time.sleep(60)",
                "description": "Timeout test",
                "timeout_seconds": 1,
            },
            context,
        )
        assert "TIMED OUT" in result.output_text

    async def test_policy_timeout_takes_precedence(self, tool: ExecuteCodeTool) -> None:
        ctx = ExecutionContext(max_execution_time_seconds=1)
        result = await tool.execute(
            {
                "code": "import time; time.sleep(60)",
                "description": "Policy timeout test",
                "timeout_seconds": 300,
            },
            ctx,
        )
        assert "TIMED OUT" in result.output_text

    async def test_missing_code_raises(
        self, tool: ExecuteCodeTool, context: ExecutionContext
    ) -> None:
        with pytest.raises(ToolInputValidationError, match="code"):
            await tool.execute({"description": "No code"}, context)

    async def test_missing_description_raises(
        self, tool: ExecuteCodeTool, context: ExecutionContext
    ) -> None:
        with pytest.raises(ToolInputValidationError, match="description"):
            await tool.execute({"code": "print(1)"}, context)

    async def test_working_directory_from_context(
        self, tool: ExecuteCodeTool, tmp_path: object
    ) -> None:
        ctx = ExecutionContext(working_directory=str(tmp_path))
        result = await tool.execute(
            {"code": "import os; print(os.getcwd())", "description": "CWD test"}, ctx
        )
        assert str(tmp_path) in result.output_text

    async def test_no_image_content_without_matplotlib(
        self, tool: ExecuteCodeTool, context: ExecutionContext
    ) -> None:
        result = await tool.execute(
            {"code": "print('text only')", "description": "No images"}, context
        )
        assert result.image_content is None
