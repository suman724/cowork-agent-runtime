"""Tests for DeleteFile tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_runtime.exceptions import FileNotFoundToolError, ToolExecutionError
from tool_runtime.models import ExecutionContext
from tool_runtime.platform.darwin import DarwinAdapter
from tool_runtime.tools.file.delete_file import DeleteFileTool


@pytest.fixture
def platform() -> DarwinAdapter:
    return DarwinAdapter()


@pytest.fixture
def tool(platform: DarwinAdapter) -> DeleteFileTool:
    return DeleteFileTool(platform)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestDeleteFile:
    async def test_delete_existing_file(
        self, tool: DeleteFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "delete_me.txt"
        p.write_text("content\n")

        result = await tool.execute({"path": str(p)}, context)
        assert not p.exists()
        assert "Deleted" in result.output_text

    async def test_delete_nonexistent_file(
        self, tool: DeleteFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundToolError, match="not found"):
            await tool.execute({"path": str(p)}, context)

    async def test_delete_directory_rejected(
        self, tool: DeleteFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        d = tmp_path / "subdir"
        d.mkdir()
        with pytest.raises(ToolExecutionError, match="directory"):
            await tool.execute({"path": str(d)}, context)
