"""Tests for ViewImage tool."""

from __future__ import annotations

import base64
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
from tool_runtime.tools.file.view_image import ViewImageTool


@pytest.fixture
def platform() -> DarwinAdapter:
    return DarwinAdapter()


@pytest.fixture
def tool(platform: DarwinAdapter) -> ViewImageTool:
    return ViewImageTool(platform)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


# Minimal valid PNG header (1x1 pixel)
_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"  # PNG signature
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Minimal valid JPEG
_MINIMAL_JPEG = bytes(
    [
        0xFF,
        0xD8,
        0xFF,
        0xE0,
        0x00,
        0x10,
        0x4A,
        0x46,
        0x49,
        0x46,
        0x00,
        0x01,
        0x01,
        0x00,
        0x00,
        0x01,
        0x00,
        0x01,
        0x00,
        0x00,
        0xFF,
        0xD9,
    ]
)


class TestViewImage:
    async def test_read_png(
        self, tool: ViewImageTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "test.png"
        p.write_bytes(_MINIMAL_PNG)

        result = await tool.execute({"path": str(p)}, context)
        assert "Image:" in result.output_text
        assert result.image_content is not None
        assert result.image_content.media_type == "image/png"
        # Verify base64 is valid
        decoded = base64.b64decode(result.image_content.base64_data)
        assert decoded == _MINIMAL_PNG

    async def test_read_jpeg(
        self, tool: ViewImageTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "photo.jpg"
        p.write_bytes(_MINIMAL_JPEG)

        result = await tool.execute({"path": str(p)}, context)
        assert result.image_content is not None
        assert result.image_content.media_type == "image/jpeg"

    async def test_read_jpeg_extension(
        self, tool: ViewImageTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "photo.jpeg"
        p.write_bytes(_MINIMAL_JPEG)

        result = await tool.execute({"path": str(p)}, context)
        assert result.image_content is not None
        assert result.image_content.media_type == "image/jpeg"

    async def test_unsupported_extension(
        self, tool: ViewImageTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.svg"
        p.write_bytes(b"<svg></svg>")

        with pytest.raises(ToolInputValidationError, match="Unsupported"):
            await tool.execute({"path": str(p)}, context)

    async def test_file_not_found(
        self, tool: ViewImageTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundToolError):
            await tool.execute({"path": str(tmp_path / "missing.png")}, context)

    async def test_file_too_large(self, tool: ViewImageTool, tmp_path: Path) -> None:
        ctx = ExecutionContext(max_file_size_bytes=100)
        p = tmp_path / "big.png"
        p.write_bytes(b"\x89PNG" + b"\x00" * 200)

        with pytest.raises(FileTooLargeError):
            await tool.execute({"path": str(p)}, ctx)

    async def test_not_a_file(
        self, tool: ViewImageTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        d = tmp_path / "dir.png"
        d.mkdir()
        with pytest.raises(ToolExecutionError, match="not a file"):
            await tool.execute({"path": str(d)}, context)

    async def test_output_includes_size(
        self, tool: ViewImageTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "test.png"
        p.write_bytes(_MINIMAL_PNG)

        result = await tool.execute({"path": str(p)}, context)
        # Should include human-readable size
        assert "B)" in result.output_text or "KB)" in result.output_text

    async def test_gif_format(
        self, tool: ViewImageTool, context: ExecutionContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "anim.gif"
        # Minimal GIF header
        p.write_bytes(
            b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
        )

        result = await tool.execute({"path": str(p)}, context)
        assert result.image_content is not None
        assert result.image_content.media_type == "image/gif"
