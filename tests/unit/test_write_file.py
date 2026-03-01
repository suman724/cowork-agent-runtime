"""Tests for WriteFile tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_runtime.models import ExecutionContext
from tool_runtime.platform.darwin import DarwinAdapter
from tool_runtime.tools.file.write_file import WriteFileTool


@pytest.fixture
def platform() -> DarwinAdapter:
    return DarwinAdapter()


@pytest.fixture
def tool(platform: DarwinAdapter) -> WriteFileTool:
    return WriteFileTool(platform)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestWriteFile:
    async def test_create_new_file(
        self, tool: WriteFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "new.txt"
        result = await tool.execute({"path": str(p), "content": "Hello, world!\n"}, context)
        assert p.read_text() == "Hello, world!\n"
        assert "Created" in result.output_text

    async def test_overwrite_existing_file(
        self, tool: WriteFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "existing.txt"
        p.write_text("old content\n")

        result = await tool.execute({"path": str(p), "content": "new content\n"}, context)
        assert p.read_text() == "new content\n"
        assert "Updated" in result.output_text

    async def test_creates_parent_directories(
        self, tool: WriteFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "sub" / "dir" / "file.txt"
        result = await tool.execute({"path": str(p), "content": "nested\n"}, context)
        assert p.read_text() == "nested\n"
        assert "Created" in result.output_text

    async def test_diff_in_output(
        self, tool: WriteFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "diff.txt"
        p.write_text("line1\nline2\n")

        result = await tool.execute({"path": str(p), "content": "line1\nline3\n"}, context)
        assert "-line2" in result.output_text
        assert "+line3" in result.output_text

    async def test_atomic_write(
        self, tool: WriteFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "atomic.txt"
        p.write_text("original\n")

        # After successful write, no temp files should remain
        await tool.execute({"path": str(p), "content": "updated\n"}, context)

        assert p.read_text() == "updated\n"
        # Check no temp files left behind
        temp_files = [f for f in tmp_path.iterdir() if f.name.startswith(".write_")]
        assert len(temp_files) == 0

    async def test_symlink_resolved_on_overwrite(
        self, tool: WriteFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        """When overwriting a symlink, the real target file should be updated."""
        real = tmp_path / "real.txt"
        real.write_text("original\n")
        link = tmp_path / "link.txt"
        link.symlink_to(real)

        await tool.execute({"path": str(link), "content": "updated\n"}, context)

        # The real file should contain the new content
        assert real.read_text() == "updated\n"
