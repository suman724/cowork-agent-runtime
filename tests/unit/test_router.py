"""Tests for ToolRouter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_tool_request
from tool_runtime.models import ExecutionContext
from tool_runtime.router.tool_router import ToolRouter


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestToolRouter:
    async def test_dispatch_read_file(
        self, tool_router: ToolRouter, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "test.txt"
        p.write_text("hello\n")

        request = make_tool_request("ReadFile", {"path": str(p)})
        result = await tool_router.execute(request, context)

        assert result.tool_result.status == "succeeded"
        assert result.tool_result.outputText is not None
        assert "hello" in result.tool_result.outputText

    async def test_unknown_tool_returns_failed(
        self, tool_router: ToolRouter, context: ExecutionContext
    ) -> None:
        request = make_tool_request("NonExistentTool", {})
        result = await tool_router.execute(request, context)

        assert result.tool_result.status == "failed"
        assert result.tool_result.error is not None
        assert result.tool_result.error.code == "TOOL_NOT_FOUND"

    async def test_validation_error_returns_failed(
        self, tool_router: ToolRouter, context: ExecutionContext
    ) -> None:
        # ReadFile requires 'path' argument
        request = make_tool_request("ReadFile", {})
        result = await tool_router.execute(request, context)

        assert result.tool_result.status == "failed"
        assert result.tool_result.error is not None
        assert result.tool_result.error.code == "INVALID_REQUEST"

    async def test_get_available_tools(self, tool_router: ToolRouter) -> None:
        tools = tool_router.get_available_tools()
        tool_names = {t.toolName for t in tools}

        assert tool_names == {"ReadFile", "WriteFile", "DeleteFile", "RunCommand", "HttpRequest"}
        for tool_def in tools:
            assert tool_def.description
            assert tool_def.inputSchema

    async def test_result_carries_session_ids(
        self, tool_router: ToolRouter, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "ids.txt"
        p.write_text("test\n")

        request = make_tool_request(
            "ReadFile",
            {"path": str(p)},
            session_id="s1",
            task_id="t1",
            step_id="st1",
        )
        result = await tool_router.execute(request, context)

        assert result.tool_result.sessionId == "s1"
        assert result.tool_result.taskId == "t1"
        assert result.tool_result.stepId == "st1"

    async def test_default_context_when_none(self, tool_router: ToolRouter, tmp_path: Path) -> None:
        p = tmp_path / "default_ctx.txt"
        p.write_text("works\n")

        request = make_tool_request("ReadFile", {"path": str(p)})
        result = await tool_router.execute(request, None)

        assert result.tool_result.status == "succeeded"

    async def test_unexpected_error_captured(
        self, tool_router: ToolRouter, context: ExecutionContext
    ) -> None:
        """Unexpected exceptions are caught and returned as failed results."""
        request = make_tool_request("ReadFile", {"path": "/tmp/test.txt"})

        # Mock the tool to raise an unexpected error
        mock_tool = AsyncMock()
        mock_tool.name = "ReadFile"
        mock_tool.validate_input = lambda _args: None
        mock_tool.execute = AsyncMock(side_effect=RuntimeError("kaboom"))
        tool_router._tools["ReadFile"] = mock_tool

        result = await tool_router.execute(request, context)

        assert result.tool_result.status == "failed"
        assert result.tool_result.error is not None
        assert result.tool_result.error.code == "TOOL_EXECUTION_FAILED"
        assert "kaboom" in (result.tool_result.error.message or "")

    async def test_artifact_in_result(
        self, tool_router: ToolRouter, context: ExecutionContext, tmp_path: Path
    ) -> None:
        """Large outputs produce artifacts in the result."""
        p = tmp_path / "big.txt"
        # Write more than ARTIFACT_THRESHOLD (10KB)
        p.write_text("x" * 15000)

        request = make_tool_request("ReadFile", {"path": str(p)})
        result = await tool_router.execute(request, context)

        assert result.tool_result.status == "succeeded"


class TestPlatformDetection:
    def test_get_platform_returns_darwin_on_mac(self) -> None:
        from tool_runtime.platform.detection import get_platform

        adapter = get_platform()
        assert adapter.shell_executable == "/bin/zsh"

    @patch("tool_runtime.platform.detection.sys")
    def test_get_platform_returns_windows_on_win32(self, mock_sys: object) -> None:
        from tool_runtime.platform.detection import get_platform

        mock_sys.platform = "win32"  # type: ignore[attr-defined]
        adapter = get_platform()
        assert adapter.shell_executable == "cmd.exe"
