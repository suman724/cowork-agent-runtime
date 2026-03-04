"""Tests for memory tools in AgentToolHandler."""

from __future__ import annotations

from pathlib import Path

from agent_host.loop.agent_tools import AgentToolHandler
from agent_host.memory.memory_manager import MemoryManager
from agent_host.memory.working_memory import WorkingMemory


class TestMemoryToolsRouting:
    def test_is_agent_tool_recognizes_memory_tools(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        assert handler.is_agent_tool("SaveMemory")
        assert handler.is_agent_tool("RecallMemory")
        assert handler.is_agent_tool("ListMemories")
        assert handler.is_agent_tool("DeleteMemory")

    def test_memory_tools_in_definitions_when_manager_present(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mm._persistent_memory._memory_dir = tmp_path / "memory"
        handler = AgentToolHandler(WorkingMemory(), memory_manager=mm)
        defs = handler.get_tool_definitions()
        names = {d["function"]["name"] for d in defs}
        assert "SaveMemory" in names
        assert "RecallMemory" in names
        assert "ListMemories" in names
        assert "DeleteMemory" in names

    def test_memory_tools_absent_without_manager(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        defs = handler.get_tool_definitions()
        names = {d["function"]["name"] for d in defs}
        assert "SaveMemory" not in names
        assert "RecallMemory" not in names
        assert "ListMemories" not in names
        assert "DeleteMemory" not in names


class TestMemoryToolExecution:
    async def test_save_memory(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        mm._persistent_memory._memory_dir = mem_dir
        handler = AgentToolHandler(WorkingMemory(), memory_manager=mm)

        result = await handler.execute("SaveMemory", {"content": "# Notes\nKey insight"})
        assert result["status"] == "success"
        assert (mem_dir / "MEMORY.md").read_text() == "# Notes\nKey insight"

    async def test_save_to_named_file(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        mm._persistent_memory._memory_dir = mem_dir
        handler = AgentToolHandler(WorkingMemory(), memory_manager=mm)

        result = await handler.execute(
            "SaveMemory", {"file": "debugging.md", "content": "Debug notes"}
        )
        assert result["status"] == "success"
        assert (mem_dir / "debugging.md").read_text() == "Debug notes"

    async def test_recall_memory(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "patterns.md").write_text("Use factory pattern")
        mm._persistent_memory._memory_dir = mem_dir
        handler = AgentToolHandler(WorkingMemory(), memory_manager=mm)

        result = await handler.execute("RecallMemory", {"file": "patterns.md"})
        assert result["status"] == "success"
        assert result["content"] == "Use factory pattern"

    async def test_list_memories(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("index")
        (mem_dir / "api.md").write_text("api notes")
        mm._persistent_memory._memory_dir = mem_dir
        handler = AgentToolHandler(WorkingMemory(), memory_manager=mm)

        result = await handler.execute("ListMemories", {})
        assert result["status"] == "success"
        names = [f["name"] for f in result["files"]]
        assert "MEMORY.md" in names
        assert "api.md" in names

    async def test_delete_memory(self, tmp_path: Path) -> None:
        mm = MemoryManager(str(tmp_path))
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "old.md").write_text("obsolete")
        mm._persistent_memory._memory_dir = mem_dir
        handler = AgentToolHandler(WorkingMemory(), memory_manager=mm)

        result = await handler.execute("DeleteMemory", {"file": "old.md"})
        assert result["status"] == "success"
        assert not (mem_dir / "old.md").exists()

    async def test_memory_tools_fail_gracefully_without_manager(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        for tool in ("SaveMemory", "RecallMemory", "ListMemories", "DeleteMemory"):
            result = await handler.execute(tool, {"content": "test", "file": "test.md"})
            assert result["status"] == "error"
