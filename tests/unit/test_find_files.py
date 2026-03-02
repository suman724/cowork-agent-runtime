"""Tests for FindFiles tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    ToolExecutionError,
    ToolInputValidationError,
)
from tool_runtime.models import ExecutionContext
from tool_runtime.tools.file.find_files import FindFilesTool


@pytest.fixture
def tool() -> FindFilesTool:
    return FindFilesTool()


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestFindFiles:
    async def test_simple_glob(
        self, tool: FindFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "foo.py").write_text("code")
        (tmp_path / "bar.txt").write_text("text")

        result = await tool.execute({"directory": str(tmp_path), "pattern": "*.py"}, context)
        assert "foo.py" in result.output_text
        assert "bar.txt" not in result.output_text

    async def test_nested_dirs(
        self, tool: FindFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        sub = tmp_path / "src" / "pkg"
        sub.mkdir(parents=True)
        (sub / "mod.py").write_text("code")
        (tmp_path / "top.py").write_text("code")

        result = await tool.execute({"directory": str(tmp_path), "pattern": "*.py"}, context)
        assert "src/pkg/mod.py" in result.output_text or "src\\pkg\\mod.py" in result.output_text
        assert "top.py" in result.output_text

    async def test_no_matches(
        self, tool: FindFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "foo.txt").write_text("text")

        result = await tool.execute({"directory": str(tmp_path), "pattern": "*.py"}, context)
        assert "No files matching" in result.output_text

    async def test_max_results_cap(
        self, tool: FindFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text(f"content {i}")

        result = await tool.execute(
            {"directory": str(tmp_path), "pattern": "*.txt", "max_results": 3}, context
        )
        assert "truncated" in result.output_text.lower()

    async def test_nonexistent_directory(
        self, tool: FindFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundToolError):
            await tool.execute({"directory": str(tmp_path / "nope"), "pattern": "*"}, context)

    async def test_not_a_directory(
        self, tool: FindFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(ToolExecutionError, match="not a directory"):
            await tool.execute({"directory": str(f), "pattern": "*"}, context)

    async def test_results_sorted(
        self, tool: FindFilesTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        (tmp_path / "c.py").write_text("c")
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")

        result = await tool.execute({"directory": str(tmp_path), "pattern": "*.py"}, context)
        lines = [line for line in result.output_text.strip().split("\n") if line]
        assert lines == sorted(lines)

    async def test_requires_absolute_path(
        self, tool: FindFilesTool, context: ExecutionContext
    ) -> None:
        with pytest.raises(ToolInputValidationError, match="absolute"):
            await tool.execute({"directory": "relative/path", "pattern": "*.py"}, context)
