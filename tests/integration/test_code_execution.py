"""Integration test: full pipeline through ToolRouter for code execution."""

from __future__ import annotations

import pytest

from tool_runtime.models import ExecutionContext
from tool_runtime.platform.detection import get_platform
from tool_runtime.router.tool_router import ToolRouter

from tests.conftest import make_tool_request


@pytest.fixture
def tool_router() -> ToolRouter:
    return ToolRouter(platform=get_platform())


@pytest.fixture
def context(tmp_path: object) -> ExecutionContext:
    return ExecutionContext(working_directory=str(tmp_path))


@pytest.mark.integration
class TestCodeExecutionIntegration:
    async def test_execute_code_through_router(
        self, tool_router: ToolRouter, context: ExecutionContext
    ) -> None:
        """Full pipeline: ToolRouter → ExecuteCodeTool → PythonExecutor → result."""
        request = make_tool_request(
            "ExecuteCode",
            arguments={
                "code": "print(sum(range(10)))",
                "description": "Sum 0..9",
            },
        )
        result = await tool_router.execute(request, context)
        assert result.tool_result.status == "succeeded"
        assert "45" in (result.tool_result.outputText or "")

    async def test_execute_code_error_through_router(
        self, tool_router: ToolRouter, context: ExecutionContext
    ) -> None:
        request = make_tool_request(
            "ExecuteCode",
            arguments={
                "code": "raise RuntimeError('test error')",
                "description": "Error test",
            },
        )
        result = await tool_router.execute(request, context)
        # Tool still succeeds (exit code captured), but output shows error
        assert result.tool_result.status == "succeeded"
        assert "RuntimeError" in (result.tool_result.outputText or "")

    async def test_execute_code_registered(self, tool_router: ToolRouter) -> None:
        """Verify ExecuteCode is in the available tools list."""
        tool_names = [t.toolName for t in tool_router.get_available_tools()]
        assert "ExecuteCode" in tool_names

    async def test_execute_code_with_working_directory(
        self, tool_router: ToolRouter, tmp_path: object
    ) -> None:
        ctx = ExecutionContext(working_directory=str(tmp_path))
        request = make_tool_request(
            "ExecuteCode",
            arguments={
                "code": "import os; print(os.getcwd())",
                "description": "Check CWD",
            },
        )
        result = await tool_router.execute(request, ctx)
        assert result.tool_result.status == "succeeded"
        assert str(tmp_path) in (result.tool_result.outputText or "")
