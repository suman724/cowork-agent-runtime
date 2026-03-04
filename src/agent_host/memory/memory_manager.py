"""MemoryManager — orchestrates project instructions and persistent memory."""

from __future__ import annotations

from typing import Any

import structlog

from agent_host.memory.persistent_memory import PersistentMemory
from agent_host.memory.project_instructions import ProjectInstructionsLoader

logger = structlog.get_logger()


class MemoryManager:
    """Orchestrates project instructions (COWORK.md) and persistent auto-memory.

    Created once per session in ``SessionManager._init_components()``.
    Provides:
      - Project instructions for system prompt injection
      - Auto-memory index for per-turn context injection
      - Tool handlers for SaveMemory / RecallMemory / ListMemories
    """

    def __init__(self, workspace_dir: str) -> None:
        self._workspace_dir = workspace_dir
        self._instructions_loader = ProjectInstructionsLoader()
        memory_dir = PersistentMemory.resolve_memory_dir(workspace_dir)
        self._persistent_memory = PersistentMemory(memory_dir)
        self._project_instructions: str = ""
        self._memory_index: str = ""

    def load_all(self) -> None:
        """Load project instructions and memory index.  Called at session init."""
        self._project_instructions = self._instructions_loader.load(self._workspace_dir)
        self._memory_index = self._persistent_memory.load_index()

        if self._project_instructions:
            logger.info(
                "project_instructions_loaded",
                length=len(self._project_instructions),
            )
        if self._memory_index:
            logger.info("memory_index_loaded", length=len(self._memory_index))

    @property
    def project_instructions(self) -> str:
        """Loaded project instructions text (for system prompt injection)."""
        return self._project_instructions

    @property
    def memory_dir(self) -> str:
        """Resolved memory directory path."""
        return str(self._persistent_memory._memory_dir)

    def render_memory_context(self) -> str:
        """Render auto-memory for injection as a system message.

        Re-reads MEMORY.md each turn so the agent sees its own updates
        within the same session.
        """
        self._memory_index = self._persistent_memory.load_index()
        if not self._memory_index:
            return ""
        return f"# Persistent Memory\n\n{self._memory_index}"

    # ------------------------------------------------------------------
    # Tool handlers (called by AgentToolHandler)
    # ------------------------------------------------------------------

    def handle_save_memory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle SaveMemory tool call."""
        filename = arguments.get("file", "MEMORY.md")
        content = arguments.get("content", "")

        if not content:
            return {"status": "error", "message": "content is required"}

        result = self._persistent_memory.save_file(filename, content)
        if result.startswith("Error"):
            return {"status": "error", "message": result}
        return {"status": "success", "message": result}

    def handle_recall_memory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle RecallMemory tool call."""
        filename = arguments.get("file", "")
        if not filename:
            return {"status": "error", "message": "file is required"}

        result = self._persistent_memory.read_file(filename)
        if result.startswith("Error"):
            return {"status": "error", "message": result}
        return {"status": "success", "content": result}

    def handle_list_memories(self) -> dict[str, Any]:
        """Handle ListMemories tool call."""
        files = self._persistent_memory.list_files()
        return {"status": "success", "files": files}
