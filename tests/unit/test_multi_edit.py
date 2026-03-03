"""Tests for MultiEdit tool."""

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
from tool_runtime.tools.file.multi_edit import MultiEditTool


@pytest.fixture
def platform() -> DarwinAdapter:
    return DarwinAdapter()


@pytest.fixture
def tool(platform: DarwinAdapter) -> MultiEditTool:
    return MultiEditTool(platform)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestMultiEdit:
    async def test_multiple_edits(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.py"
        p.write_text("x = 1\ny = 2\nz = 3\n")

        result = await tool.execute(
            {
                "path": str(p),
                "edits": [
                    {"old_text": "x = 1", "new_text": "x = 10"},
                    {"old_text": "z = 3", "new_text": "z = 30"},
                ],
            },
            context,
        )

        assert p.read_text() == "x = 10\ny = 2\nz = 30\n"
        assert "Edited" in result.output_text
        assert "2 edits" in result.output_text

    async def test_single_edit(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("hello world\n")

        result = await tool.execute(
            {
                "path": str(p),
                "edits": [{"old_text": "hello", "new_text": "goodbye"}],
            },
            context,
        )

        assert p.read_text() == "goodbye world\n"
        assert "1 edits" in result.output_text

    async def test_diff_output(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "code.py"
        p.write_text("a = 1\nb = 2\n")

        result = await tool.execute(
            {
                "path": str(p),
                "edits": [
                    {"old_text": "a = 1", "new_text": "a = 100"},
                    {"old_text": "b = 2", "new_text": "b = 200"},
                ],
            },
            context,
        )

        assert "-a = 1" in result.output_text
        assert "+a = 100" in result.output_text
        assert "-b = 2" in result.output_text
        assert "+b = 200" in result.output_text

    async def test_old_text_not_found(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("hello\n")

        with pytest.raises(ToolExecutionError, match=r"edits\[0\]\.old_text not found"):
            await tool.execute(
                {
                    "path": str(p),
                    "edits": [{"old_text": "not here", "new_text": "replacement"}],
                },
                context,
            )
        assert p.read_text() == "hello\n"

    async def test_duplicate_matches_error(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("foo\nfoo\n")

        with pytest.raises(ToolExecutionError, match=r"edits\[0\]\.old_text matches 2 times"):
            await tool.execute(
                {
                    "path": str(p),
                    "edits": [{"old_text": "foo", "new_text": "bar"}],
                },
                context,
            )
        assert p.read_text() == "foo\nfoo\n"

    async def test_overlapping_edits_error(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("abcdef\n")

        with pytest.raises(ToolExecutionError, match="overlapping"):
            await tool.execute(
                {
                    "path": str(p),
                    "edits": [
                        {"old_text": "abcd", "new_text": "ABCD"},
                        {"old_text": "cdef", "new_text": "CDEF"},
                    ],
                },
                context,
            )
        assert p.read_text() == "abcdef\n"

    async def test_all_no_ops(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("unchanged\n")

        result = await tool.execute(
            {
                "path": str(p),
                "edits": [{"old_text": "unchanged", "new_text": "unchanged"}],
            },
            context,
        )

        assert p.read_text() == "unchanged\n"
        assert "No changes needed" in result.output_text

    async def test_mixed_noop_and_real_edits(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("a = 1\nb = 2\n")

        result = await tool.execute(
            {
                "path": str(p),
                "edits": [
                    {"old_text": "a = 1", "new_text": "a = 1"},  # no-op
                    {"old_text": "b = 2", "new_text": "b = 99"},  # real edit
                ],
            },
            context,
        )

        assert p.read_text() == "a = 1\nb = 99\n"
        assert "1 edits" in result.output_text

    async def test_empty_edits_array(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("content\n")

        with pytest.raises(ToolInputValidationError, match="empty"):
            await tool.execute({"path": str(p), "edits": []}, context)

    async def test_file_not_found(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundToolError):
            await tool.execute(
                {
                    "path": str(tmp_path / "nope.txt"),
                    "edits": [{"old_text": "a", "new_text": "b"}],
                },
                context,
            )

    async def test_requires_absolute_path(
        self, tool: MultiEditTool, context: ExecutionContext
    ) -> None:
        with pytest.raises(ToolInputValidationError, match="absolute"):
            await tool.execute(
                {
                    "path": "relative.txt",
                    "edits": [{"old_text": "a", "new_text": "b"}],
                },
                context,
            )

    async def test_atomic_write_no_temp_files(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("old\n")

        await tool.execute(
            {
                "path": str(p),
                "edits": [{"old_text": "old", "new_text": "new"}],
            },
            context,
        )

        temp_files = [f for f in tmp_path.iterdir() if f.name.startswith(".multiedit_")]
        assert len(temp_files) == 0

    async def test_earlier_edit_introduces_duplicate_for_later_edit(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        """Earlier edit introduces text causing later edit's old_text to match >1 times."""
        p = tmp_path / "file.txt"
        p.write_text("AAA\nBBB\n")

        # First edit changes AAA → BBB, now there are two BBBs
        # Second edit tries to match BBB — should fail with 2 matches
        with pytest.raises(ToolExecutionError, match="matches 2 times after applying"):
            await tool.execute(
                {
                    "path": str(p),
                    "edits": [
                        {"old_text": "AAA", "new_text": "BBB"},
                        {"old_text": "BBB", "new_text": "CCC"},
                    ],
                },
                context,
            )

    async def test_edits_applied_in_order(
        self, tool: MultiEditTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("first\nsecond\nthird\n")

        await tool.execute(
            {
                "path": str(p),
                "edits": [
                    {"old_text": "first", "new_text": "1st"},
                    {"old_text": "second", "new_text": "2nd"},
                    {"old_text": "third", "new_text": "3rd"},
                ],
            },
            context,
        )

        assert p.read_text() == "1st\n2nd\n3rd\n"
