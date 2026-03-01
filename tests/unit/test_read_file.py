"""Tests for ReadFile tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tool_runtime.exceptions import (
    FileNotFoundToolError,
    FileTooLargeError,
    ToolExecutionError,
    ToolInputValidationError,
)
from tool_runtime.models import ExecutionContext
from tool_runtime.platform.darwin import DarwinAdapter
from tool_runtime.tools.file.read_file import ReadFileTool


@pytest.fixture
def platform() -> DarwinAdapter:
    return DarwinAdapter()


@pytest.fixture
def tool(platform: DarwinAdapter) -> ReadFileTool:
    return ReadFileTool(platform)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestReadFile:
    async def test_read_text_file(
        self, tool: ReadFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "hello.txt"
        p.write_text("Hello, world!\nLine 2\n")

        result = await tool.execute({"path": str(p)}, context)
        assert result.output_text == "Hello, world!\nLine 2\n"

    async def test_read_with_offset_and_limit(
        self, tool: ReadFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "lines.txt"
        p.write_text("line1\nline2\nline3\nline4\nline5\n")

        result = await tool.execute({"path": str(p), "offset": 2, "limit": 2}, context)
        assert result.output_text == "line2\nline3\n"

    async def test_file_not_found(
        self, tool: ReadFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundToolError, match="not found"):
            await tool.execute({"path": str(p)}, context)

    async def test_file_too_large(self, tool: ReadFileTool, tmp_path: Path) -> None:
        p = tmp_path / "large.txt"
        p.write_text("x" * 1000)

        ctx = ExecutionContext(max_file_size_bytes=500)
        with pytest.raises(FileTooLargeError, match="exceeds limit"):
            await tool.execute({"path": str(p)}, ctx)

    async def test_binary_file_detected(
        self, tool: ReadFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "binary.bin"
        p.write_bytes(b"\x00\x01\x02\x03" * 100)

        result = await tool.execute({"path": str(p)}, context)
        assert "Binary file" in result.output_text

    async def test_encoding_fallback(
        self, tool: ReadFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "latin.txt"
        # Latin-1 encoded text (not valid UTF-8)
        p.write_bytes(b"caf\xe9\n")

        result = await tool.execute({"path": str(p)}, context)
        assert "caf" in result.output_text

    async def test_directory_rejected(
        self, tool: ReadFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        with pytest.raises(ToolExecutionError, match="not a file"):
            await tool.execute({"path": str(tmp_path)}, context)

    async def test_utf8_bom_stripped(
        self, tool: ReadFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "bom.txt"
        p.write_bytes(b"\xef\xbb\xbfHello BOM\n")

        result = await tool.execute({"path": str(p)}, context)
        assert result.output_text.startswith("Hello BOM")

    async def test_offset_zero_rejected(
        self, tool: ReadFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "test.txt"
        p.write_text("line1\n")

        with pytest.raises(ToolInputValidationError, match=">= 1"):
            await tool.execute({"path": str(p), "offset": 0}, context)

    async def test_limit_zero_rejected(
        self, tool: ReadFileTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "test.txt"
        p.write_text("line1\n")

        with pytest.raises(ToolInputValidationError, match=">= 1"):
            await tool.execute({"path": str(p), "limit": 0}, context)
