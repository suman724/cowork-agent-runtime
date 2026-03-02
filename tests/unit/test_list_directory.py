"""Tests for ListDirectory tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    ToolExecutionError,
    ToolInputValidationError,
)
from tool_runtime.models import ExecutionContext
from tool_runtime.platform.darwin import DarwinAdapter
from tool_runtime.tools.file.list_directory import ListDirectoryTool


@pytest.fixture
def platform() -> DarwinAdapter:
    return DarwinAdapter()


@pytest.fixture
def tool(platform: DarwinAdapter) -> ListDirectoryTool:
    return ListDirectoryTool(platform)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestListDirectory:
    async def test_list_files_and_dirs(
        self, tool: ListDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "file.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()

        result = await tool.execute({"path": str(tmp_path)}, context)
        assert "f file.txt" in result.output_text
        assert "d subdir" in result.output_text

    async def test_empty_directory(
        self, tool: ListDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        result = await tool.execute({"path": str(tmp_path)}, context)
        assert "empty" in result.output_text.lower()

    async def test_hidden_files_excluded_by_default(
        self, tool: ListDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("public")

        result = await tool.execute({"path": str(tmp_path)}, context)
        assert ".hidden" not in result.output_text
        assert "visible.txt" in result.output_text

    async def test_hidden_files_included_when_requested(
        self, tool: ListDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("public")

        result = await tool.execute({"path": str(tmp_path), "include_hidden": True}, context)
        assert ".hidden" in result.output_text
        assert "visible.txt" in result.output_text

    async def test_nonexistent_directory(
        self, tool: ListDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundToolError):
            await tool.execute({"path": str(tmp_path / "nope")}, context)

    async def test_not_a_directory(
        self, tool: ListDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ToolExecutionError, match="not a directory"):
            await tool.execute({"path": str(f)}, context)

    async def test_file_sizes_shown(
        self, tool: ListDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "small.txt").write_text("hi")
        result = await tool.execute({"path": str(tmp_path)}, context)
        assert "f small.txt (" in result.output_text

    async def test_requires_absolute_path(
        self, tool: ListDirectoryTool, context: ExecutionContext
    ) -> None:
        with pytest.raises(ToolInputValidationError, match="absolute"):
            await tool.execute({"path": "relative/path"}, context)

    async def test_sorted_entries(
        self, tool: ListDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "c.txt").write_text("c")

        result = await tool.execute({"path": str(tmp_path)}, context)
        lines = result.output_text.strip().split("\n")
        names = [line.split(" ")[1] for line in lines]
        assert names == sorted(names)
