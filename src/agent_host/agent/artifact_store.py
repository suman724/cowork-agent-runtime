"""PendingArtifactStore — FIFO bridge for artifacts between tool_fn and after_tool_callback."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tool_runtime.models import ArtifactData


class PendingArtifactStore:
    """Thread-safe FIFO bridge for passing artifacts from ``tool_fn`` to ``after_tool_callback``.

    ``tool_fn`` produces ``ArtifactData`` from ``ToolExecutionResult`` and stores
    them here.  ``after_tool_callback`` pops and uploads them via ``WorkspaceClient``.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[str, list[ArtifactData]]] = []

    def store(self, tool_name: str, artifacts: list[ArtifactData]) -> None:
        """Store artifacts produced by a tool execution."""
        if artifacts:
            self._pending.append((tool_name, artifacts))

    def pop(self, tool_name: str) -> list[ArtifactData]:
        """Pop artifacts for a given tool name (first match). Returns empty list if none."""
        for i, (name, artifacts) in enumerate(self._pending):
            if name == tool_name:
                self._pending.pop(i)
                return artifacts
        return []
