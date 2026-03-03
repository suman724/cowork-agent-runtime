"""Tests for CreateDirectory tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_runtime.exceptions import (
    ToolExecutionError,
    ToolInputValidationError,
)
from tool_runtime.models import ExecutionContext
from tool_runtime.platform.darwin import DarwinAdapter
from tool_runtime.tools.file.create_directory import CreateDirectoryTool


@pytest.fixture
def platform() -> DarwinAdapter:
    return DarwinAdapter()


@pytest.fixture
def tool(platform: DarwinAdapter) -> CreateDirectoryTool:
    return CreateDirectoryTool(platform)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestCreateDirectory:
    async def test_create_single_directory(
        self, tool: CreateDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        target = tmp_path / "new_dir"
        result = await tool.execute({"path": str(target)}, context)

        assert target.is_dir()
        assert "Created directory" in result.output_text

    async def test_create_with_parents(
        self, tool: CreateDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        target = tmp_path / "a" / "b" / "c"
        result = await tool.execute({"path": str(target)}, context)

        assert target.is_dir()
        assert "Created directory" in result.output_text

    async def test_already_exists_directory(
        self, tool: CreateDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        target = tmp_path / "existing"
        target.mkdir()

        result = await tool.execute({"path": str(target)}, context)
        assert "already exists" in result.output_text

    async def test_already_exists_file(
        self, tool: CreateDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        target = tmp_path / "file.txt"
        target.write_text("content")

        with pytest.raises(ToolExecutionError, match="is a file"):
            await tool.execute({"path": str(target)}, context)

    async def test_parent_missing_without_create_parents(
        self, tool: CreateDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        target = tmp_path / "nonexistent" / "child"

        with pytest.raises(ToolExecutionError, match="Parent directory does not exist"):
            await tool.execute({"path": str(target), "create_parents": False}, context)

    async def test_requires_absolute_path(
        self, tool: CreateDirectoryTool, context: ExecutionContext
    ) -> None:
        with pytest.raises(ToolInputValidationError, match="absolute"):
            await tool.execute({"path": "relative/path"}, context)

    async def test_default_create_parents_is_true(
        self, tool: CreateDirectoryTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        target = tmp_path / "deep" / "nested" / "dir"
        await tool.execute({"path": str(target)}, context)
        assert target.is_dir()
