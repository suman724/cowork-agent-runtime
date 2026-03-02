"""Tests for EditFile tool."""

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
from tool_runtime.tools.file.edit_file import EditFileTool


@pytest.fixture
def platform() -> DarwinAdapter:
    return DarwinAdapter()


@pytest.fixture
def tool(platform: DarwinAdapter) -> EditFileTool:
    return EditFileTool(platform)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestEditFile:
    async def test_single_replacement(
        self, tool: EditFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.py"
        p.write_text("def foo():\n    return 1\n")

        result = await tool.execute(
            {"path": str(p), "old_text": "return 1", "new_text": "return 42"},
            context,
        )
        assert p.read_text() == "def foo():\n    return 42\n"
        assert "Edited" in result.output_text
        assert "-    return 1" in result.output_text
        assert "+    return 42" in result.output_text

    async def test_old_text_not_found(
        self, tool: EditFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("hello world\n")

        with pytest.raises(ToolExecutionError, match="old_text not found"):
            await tool.execute(
                {"path": str(p), "old_text": "not here", "new_text": "replacement"},
                context,
            )
        # File should be unchanged
        assert p.read_text() == "hello world\n"

    async def test_multiple_matches_error(
        self, tool: EditFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("foo\nfoo\nfoo\n")

        with pytest.raises(ToolExecutionError, match="matches 3 times"):
            await tool.execute(
                {"path": str(p), "old_text": "foo", "new_text": "bar"},
                context,
            )
        # File should be unchanged
        assert p.read_text() == "foo\nfoo\nfoo\n"

    async def test_empty_new_text_for_deletion(
        self, tool: EditFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("keep this\nremove this line\nkeep this too\n")

        result = await tool.execute(
            {"path": str(p), "old_text": "remove this line\n", "new_text": ""},
            context,
        )
        assert p.read_text() == "keep this\nkeep this too\n"
        assert "Edited" in result.output_text

    async def test_diff_in_output(
        self, tool: EditFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "code.py"
        p.write_text("x = 1\ny = 2\n")

        result = await tool.execute(
            {"path": str(p), "old_text": "x = 1", "new_text": "x = 100"},
            context,
        )
        assert "-x = 1" in result.output_text
        assert "+x = 100" in result.output_text

    async def test_file_not_found(
        self, tool: EditFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundToolError):
            await tool.execute(
                {"path": str(tmp_path / "nope.txt"), "old_text": "a", "new_text": "b"},
                context,
            )

    async def test_identical_text_rejected(
        self, tool: EditFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("hello\n")

        with pytest.raises(ToolInputValidationError, match="identical"):
            await tool.execute(
                {"path": str(p), "old_text": "hello", "new_text": "hello"},
                context,
            )

    async def test_atomic_write_no_temp_files(
        self, tool: EditFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("old content\n")

        await tool.execute(
            {"path": str(p), "old_text": "old content", "new_text": "new content"},
            context,
        )
        temp_files = [f for f in tmp_path.iterdir() if f.name.startswith(".edit_")]
        assert len(temp_files) == 0

    async def test_requires_absolute_path(
        self, tool: EditFileTool, context: ExecutionContext
    ) -> None:
        with pytest.raises(ToolInputValidationError, match="absolute"):
            await tool.execute(
                {"path": "relative.txt", "old_text": "a", "new_text": "b"},
                context,
            )

    async def test_multiline_replacement(
        self, tool: EditFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.py"
        p.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")

        await tool.execute(
            {
                "path": str(p),
                "old_text": "def foo():\n    pass",
                "new_text": "def foo():\n    return 42",
            },
            context,
        )
        content = p.read_text()
        assert "return 42" in content
        assert "def bar():\n    pass" in content
