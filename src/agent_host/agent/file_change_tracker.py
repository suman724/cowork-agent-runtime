"""FileChangeTracker — records file mutations for patch preview."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Literal


@dataclass
class FileChange:
    """A single file mutation recorded during a task."""

    path: str
    status: Literal["added", "modified", "deleted"]
    old_content: str  # empty string for new files
    new_content: str  # empty string for deleted files


class FileChangeTracker:
    """Tracks file changes (write / delete) per task for patch preview.

    ``tool_fn`` calls ``record_write`` / ``record_delete`` around each
    file-mutating tool execution.  ``SessionManager.get_patch_preview``
    delegates here to produce unified diffs.
    """

    def __init__(self) -> None:
        self._changes: dict[str, list[FileChange]] = {}  # task_id -> changes

    def record_write(
        self,
        task_id: str,
        path: str,
        old_content: str | None,
        new_content: str,
    ) -> None:
        """Record a file write (create or overwrite)."""
        status: Literal["added", "modified"] = "added" if old_content is None else "modified"
        change = FileChange(
            path=path,
            status=status,
            old_content=old_content or "",
            new_content=new_content,
        )
        self._changes.setdefault(task_id, []).append(change)

    def record_delete(self, task_id: str, path: str, old_content: str) -> None:
        """Record a file deletion."""
        change = FileChange(
            path=path,
            status="deleted",
            old_content=old_content,
            new_content="",
        )
        self._changes.setdefault(task_id, []).append(change)

    def get_patch_preview(self, task_id: str) -> dict[str, object]:
        """Generate a patch preview with unified diffs for a task.

        Returns:
            ``{"taskId": ..., "files": [{"filePath": ..., "status": ..., "hunks": ...}, ...]}``
        """
        changes = self._changes.get(task_id, [])

        # Deduplicate: keep the latest change per file path
        seen: dict[str, FileChange] = {}
        for change in changes:
            seen[change.path] = change

        files: list[dict[str, str]] = []
        for change in seen.values():
            diff_lines = difflib.unified_diff(
                change.old_content.splitlines(keepends=True),
                change.new_content.splitlines(keepends=True),
                fromfile=f"a/{change.path}",
                tofile=f"b/{change.path}",
            )
            hunks = "".join(diff_lines)
            files.append(
                {
                    "filePath": change.path,
                    "status": change.status,
                    "hunks": hunks,
                }
            )

        return {"taskId": task_id, "files": files}

    def clear_task(self, task_id: str) -> None:
        """Remove all tracked changes for a task."""
        self._changes.pop(task_id, None)
