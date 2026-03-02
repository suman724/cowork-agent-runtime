"""TaskTracker — structured task list the agent maintains during work."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class TrackedTask:
    """A single task tracked by the agent."""

    id: str
    content: str
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"


class TaskTracker:
    """Maintains a structured task list for goal tracking.

    The LLM can create, update, and query tasks to maintain focus
    during multi-step workflows.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TrackedTask] = {}

    @property
    def tasks(self) -> list[TrackedTask]:
        """Return all tasks in insertion order."""
        return list(self._tasks.values())

    def create_task(self, content: str) -> str:
        """Create a new task and return its ID."""
        task_id = f"t-{uuid.uuid4().hex[:8]}"
        self._tasks[task_id] = TrackedTask(id=task_id, content=content)
        return task_id

    def update_task(
        self,
        task_id: str,
        status: str | None = None,
        content: str | None = None,
    ) -> bool:
        """Update a task's status or content. Returns False if not found."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if status is not None:
            task.status = status  # type: ignore[assignment]
        if content is not None:
            task.content = content
        return True

    def get_task(self, task_id: str) -> TrackedTask | None:
        """Get a single task by ID."""
        return self._tasks.get(task_id)

    def render(self) -> str:
        """Render task list as text for injection into system prompt."""
        if not self._tasks:
            return ""
        lines = ["## Current Tasks"]
        for task in self._tasks.values():
            lines.append(f"- [{task.status}] {task.content} (id: {task.id})")
        return "\n".join(lines)

    def to_checkpoint(self) -> list[dict[str, Any]]:
        """Serialize for checkpoint persistence."""
        return [
            {"id": task.id, "content": task.content, "status": task.status}
            for task in self._tasks.values()
        ]

    @classmethod
    def from_checkpoint(cls, data: list[dict[str, Any]]) -> TaskTracker:
        """Restore from checkpoint data."""
        tracker = cls()
        for item in data:
            task = TrackedTask(
                id=item["id"],
                content=item["content"],
                status=item.get("status", "pending"),
            )
            tracker._tasks[task.id] = task
        return tracker
