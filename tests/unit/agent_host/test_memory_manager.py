"""Tests for MemoryManager — orchestrates project instructions + persistent memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_host.memory.memory_manager import MemoryManager


class TestMemoryManagerLoadAll:
    def test_loads_instructions_and_index(self, tmp_path: Path) -> None:
        # Set up project instructions
        (tmp_path / "COWORK.md").write_text("Build with pytest")

        # Set up memory directory (override resolve to use tmp_path)
        mm = MemoryManager(str(tmp_path))
        # Manually set the persistent memory dir to tmp for testing
        mm._persistent_memory._memory_dir = tmp_path / "memory"
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "MEMORY.md").write_text("# Memory\nUser prefers dark mode")

        mm.load_all()

        assert "Build with pytest" in mm.project_instructions
        assert "dark mode" in mm.render_memory_context()

    def test_empty_workspace(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mm._persistent_memory._memory_dir = tmp_path / "memory"
        mm.load_all()

        assert mm.project_instructions == ""
        assert mm.render_memory_context() == ""


class TestRenderMemoryContext:
    def test_with_content(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("Important fact")
        mm._persistent_memory._memory_dir = mem_dir

        result = mm.render_memory_context()
        assert result.startswith("# Persistent Memory")
        assert "Important fact" in result

    def test_empty_returns_empty_string(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mm._persistent_memory._memory_dir = tmp_path / "memory"
        assert mm.render_memory_context() == ""

    def test_re_reads_on_each_call(self, tmp_path: Path) -> None:
        """render_memory_context re-reads MEMORY.md so in-session saves are visible."""
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        mm._persistent_memory._memory_dir = mem_dir

        # First call — no file
        assert mm.render_memory_context() == ""

        # Create file mid-session
        (mem_dir / "MEMORY.md").write_text("New fact")
        result = mm.render_memory_context()
        assert "New fact" in result


class TestHandleSaveMemory:
    @pytest.mark.asyncio
    async def test_save_memory(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        mm._persistent_memory._memory_dir = mem_dir

        result = await mm.handle_save_memory({"content": "# Notes\nSome content"})
        assert result["status"] == "success"
        assert "MEMORY.md" in result["message"]

    @pytest.mark.asyncio
    async def test_save_to_named_file(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        mm._persistent_memory._memory_dir = mem_dir

        result = await mm.handle_save_memory({"file": "debugging.md", "content": "Debug notes"})
        assert result["status"] == "success"
        assert (mem_dir / "debugging.md").read_text() == "Debug notes"

    @pytest.mark.asyncio
    async def test_save_empty_content_rejected(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        result = await mm.handle_save_memory({"content": ""})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_save_invalid_filename(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        result = await mm.handle_save_memory({"file": "../escape.md", "content": "bad"})
        assert result["status"] == "error"


class TestHandleRecallMemory:
    @pytest.mark.asyncio
    async def test_recall_existing_file(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "patterns.md").write_text("Use factory pattern")
        mm._persistent_memory._memory_dir = mem_dir

        result = await mm.handle_recall_memory({"file": "patterns.md"})
        assert result["status"] == "success"
        assert result["content"] == "Use factory pattern"

    @pytest.mark.asyncio
    async def test_recall_missing_file(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mm._persistent_memory._memory_dir = tmp_path / "memory"
        result = await mm.handle_recall_memory({"file": "nonexistent.md"})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_recall_missing_filename(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        result = await mm.handle_recall_memory({})
        assert result["status"] == "error"


class TestHandleDeleteMemory:
    @pytest.mark.asyncio
    async def test_delete_success(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "old.md").write_text("obsolete")
        mm._persistent_memory._memory_dir = mem_dir

        result = await mm.handle_delete_memory({"file": "old.md"})
        assert result["status"] == "success"
        assert "Deleted" in result["message"]
        assert not (mem_dir / "old.md").exists()

    @pytest.mark.asyncio
    async def test_delete_missing_file(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mm._persistent_memory._memory_dir = tmp_path / "memory"
        result = await mm.handle_delete_memory({"file": "nonexistent.md"})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_delete_missing_filename(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        result = await mm.handle_delete_memory({})
        assert result["status"] == "error"
        assert "file is required" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_memory_md_rejected(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("index")
        mm._persistent_memory._memory_dir = mem_dir

        result = await mm.handle_delete_memory({"file": "MEMORY.md"})
        assert result["status"] == "error"
        assert "Cannot delete" in result["message"]


class TestHandleListMemories:
    @pytest.mark.asyncio
    async def test_list_memories(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("index")
        (mem_dir / "api.md").write_text("api notes")
        mm._persistent_memory._memory_dir = mem_dir

        result = await mm.handle_list_memories()
        assert result["status"] == "success"
        names = [f["name"] for f in result["files"]]
        assert "MEMORY.md" in names
        assert "api.md" in names

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mm._persistent_memory._memory_dir = tmp_path / "memory"
        result = await mm.handle_list_memories()
        assert result["status"] == "success"
        assert result["files"] == []


class TestStructuredErrorHandling:
    @pytest.mark.asyncio
    async def test_no_false_positive_on_error_content(self, tmp_path: Path) -> None:
        """File content starting with 'Error' should not be treated as an error."""
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        mm._persistent_memory._memory_dir = mem_dir

        # Save a file whose content starts with "Error"
        await mm.handle_save_memory({"file": "errors.md", "content": "Error handling notes"})

        # Read it back — should succeed, not be treated as error
        result = await mm.handle_recall_memory({"file": "errors.md"})
        assert result["status"] == "success"
        assert result["content"] == "Error handling notes"
