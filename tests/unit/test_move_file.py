"""Tests for MoveFile tool."""

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
from tool_runtime.tools.file.move_file import MoveFileTool


@pytest.fixture
def platform() -> DarwinAdapter:
    return DarwinAdapter()


@pytest.fixture
def tool(platform: DarwinAdapter) -> MoveFileTool:
    return MoveFileTool(platform)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestMoveFile:
    async def test_move_file(
        self, tool: MoveFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        src = tmp_path / "source.txt"
        src.write_text("content")
        dest = tmp_path / "dest.txt"

        result = await tool.execute({"source": str(src), "destination": str(dest)}, context)

        assert not src.exists()
        assert dest.read_text() == "content"
        assert "Moved" in result.output_text
        assert "\u2192" in result.output_text

    async def test_rename_file(
        self, tool: MoveFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        src = tmp_path / "old_name.py"
        src.write_text("def hello(): pass")
        dest = tmp_path / "new_name.py"

        await tool.execute({"source": str(src), "destination": str(dest)}, context)

        assert not src.exists()
        assert dest.read_text() == "def hello(): pass"

    async def test_overwrite_false_dest_exists(
        self, tool: MoveFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        src = tmp_path / "src.txt"
        src.write_text("new")
        dest = tmp_path / "dest.txt"
        dest.write_text("old")

        with pytest.raises(ToolExecutionError, match="already exists"):
            await tool.execute({"source": str(src), "destination": str(dest)}, context)
        # Source should still exist, dest unchanged
        assert src.read_text() == "new"
        assert dest.read_text() == "old"

    async def test_overwrite_true(
        self, tool: MoveFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        src = tmp_path / "src.txt"
        src.write_text("new content")
        dest = tmp_path / "dest.txt"
        dest.write_text("old content")

        await tool.execute(
            {"source": str(src), "destination": str(dest), "overwrite": True}, context
        )

        assert not src.exists()
        assert dest.read_text() == "new content"

    async def test_source_not_found(
        self, tool: MoveFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundToolError, match="not found"):
            await tool.execute(
                {
                    "source": str(tmp_path / "nope.txt"),
                    "destination": str(tmp_path / "dest.txt"),
                },
                context,
            )

    async def test_cross_directory_move(
        self, tool: MoveFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        src_dir = tmp_path / "src_dir"
        src_dir.mkdir()
        src = src_dir / "file.txt"
        src.write_text("data")

        dest_dir = tmp_path / "dest_dir"
        dest_dir.mkdir()
        dest = dest_dir / "file.txt"

        await tool.execute({"source": str(src), "destination": str(dest)}, context)

        assert not src.exists()
        assert dest.read_text() == "data"

    async def test_destination_parent_created(
        self, tool: MoveFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        src = tmp_path / "file.txt"
        src.write_text("content")
        dest = tmp_path / "new" / "nested" / "dir" / "file.txt"

        await tool.execute({"source": str(src), "destination": str(dest)}, context)

        assert not src.exists()
        assert dest.read_text() == "content"

    async def test_move_directory(
        self, tool: MoveFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        src = tmp_path / "mydir"
        src.mkdir()
        (src / "a.txt").write_text("a")
        dest = tmp_path / "renamed_dir"

        await tool.execute({"source": str(src), "destination": str(dest)}, context)

        assert not src.exists()
        assert (dest / "a.txt").read_text() == "a"

    async def test_requires_absolute_source(
        self, tool: MoveFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        with pytest.raises(ToolInputValidationError, match="absolute"):
            await tool.execute(
                {"source": "relative.txt", "destination": str(tmp_path / "dest.txt")},
                context,
            )

    async def test_requires_absolute_destination(
        self, tool: MoveFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        src = tmp_path / "file.txt"
        src.write_text("x")
        with pytest.raises(ToolInputValidationError, match="absolute"):
            await tool.execute(
                {"source": str(src), "destination": "relative.txt"},
                context,
            )
