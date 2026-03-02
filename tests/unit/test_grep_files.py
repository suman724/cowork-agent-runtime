"""Tests for GrepFiles tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    ToolExecutionError,
    ToolInputValidationError,
)
from tool_runtime.models import ExecutionContext
from tool_runtime.tools.file.grep_files import GrepFilesTool


@pytest.fixture
def tool() -> GrepFilesTool:
    return GrepFilesTool()


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestGrepFiles:
    async def test_simple_pattern(
        self, tool: GrepFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "hello.py").write_text("print('hello world')\nprint('goodbye')\n")

        result = await tool.execute({"directory": str(tmp_path), "pattern": "hello"}, context)
        assert "hello.py:1:" in result.output_text
        assert "hello world" in result.output_text

    async def test_regex_pattern(
        self, tool: GrepFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "code.py").write_text("def foo():\n    x = 42\ndef bar():\n    pass\n")

        result = await tool.execute(
            {"directory": str(tmp_path), "pattern": r"def \w+\(\)"}, context
        )
        assert "code.py:1:" in result.output_text
        assert "code.py:3:" in result.output_text

    async def test_context_lines(
        self, tool: GrepFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "log.txt").write_text("line1\nline2\nERROR here\nline4\nline5\n")

        result = await tool.execute(
            {"directory": str(tmp_path), "pattern": "ERROR", "context_lines": 1}, context
        )
        assert "line2" in result.output_text
        assert "ERROR here" in result.output_text
        assert "line4" in result.output_text

    async def test_binary_files_skipped(
        self, tool: GrepFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02hello\x03")
        (tmp_path / "text.txt").write_text("hello world\n")

        result = await tool.execute({"directory": str(tmp_path), "pattern": "hello"}, context)
        assert "text.txt" in result.output_text
        assert "binary.bin" not in result.output_text

    async def test_invalid_regex(
        self, tool: GrepFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "file.txt").write_text("test\n")
        with pytest.raises(ToolInputValidationError, match="Invalid regex"):
            await tool.execute({"directory": str(tmp_path), "pattern": "[invalid"}, context)

    async def test_no_matches(
        self, tool: GrepFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "file.txt").write_text("nothing here\n")

        result = await tool.execute({"directory": str(tmp_path), "pattern": "notfound"}, context)
        assert "No matches" in result.output_text

    async def test_file_glob_filter(
        self, tool: GrepFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "code.py").write_text("hello\n")
        (tmp_path / "readme.md").write_text("hello\n")

        result = await tool.execute(
            {"directory": str(tmp_path), "pattern": "hello", "file_glob": "*.py"}, context
        )
        assert "code.py" in result.output_text
        assert "readme.md" not in result.output_text

    async def test_max_results_truncation(
        self, tool: GrepFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        lines = "\n".join(f"match line {i}" for i in range(20))
        (tmp_path / "big.txt").write_text(lines)

        result = await tool.execute(
            {"directory": str(tmp_path), "pattern": "match", "max_results": 5}, context
        )
        assert "truncated" in result.output_text.lower()

    async def test_nonexistent_directory(
        self, tool: GrepFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundToolError):
            await tool.execute({"directory": str(tmp_path / "nope"), "pattern": "test"}, context)

    async def test_not_a_directory(
        self, tool: GrepFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(ToolExecutionError, match="not a directory"):
            await tool.execute({"directory": str(f), "pattern": "test"}, context)

    async def test_requires_absolute_path(
        self, tool: GrepFilesTool, context: ExecutionContext
    ) -> None:
        with pytest.raises(ToolInputValidationError, match="absolute"):
            await tool.execute({"directory": "relative", "pattern": "test"}, context)
