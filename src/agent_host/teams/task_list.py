"""SharedTaskList — in-memory task list with dependency resolution."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from agent_host.teams.models import TaskStatus, TeamTask


class SharedTaskList:
    """Thread-safe shared task list with automatic dependency unblocking.

    All mutations are protected by an asyncio.Lock for cooperative concurrency.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TeamTask] = {}
        self._lock = asyncio.Lock()

    async def create_task(
        self,
        title: str,
        description: str,
        blocked_by: list[str] | None = None,
        created_by: str = "",
    ) -> TeamTask:
        """Create a new task. Status is 'blocked' if it has unresolved dependencies."""
        blocked_by = blocked_by or []
        async with self._lock:
            if blocked_by:
                self._validate_dependencies(blocked_by)
            status: TaskStatus = "blocked" if blocked_by else "pending"
            task = TeamTask(
                title=title,
                description=description,
                status=status,
                blocked_by=list(blocked_by),
                created_by=created_by,
            )
            self._tasks[task.task_id] = task
            return task

    async def assign_task(self, task_id: str, assignee: str) -> TeamTask:
        """Assign a pending task to a teammate and set status to in_progress."""
        async with self._lock:
            task = self._get_or_raise(task_id)
            if task.status != "pending":
                msg = f"Cannot assign task {task_id}: status is '{task.status}', expected 'pending'"
                raise ValueError(msg)
            task.assignee = assignee
            task.status = "in_progress"
            task.updated_at = datetime.now(tz=UTC)
            return task

    async def update_status(
        self,
        task_id: str,
        status: str,
        result: str | None = None,
    ) -> tuple[TeamTask, list[TeamTask]]:
        """Update task status. Completing a task auto-unblocks dependents.

        Returns a tuple of (updated_task, list_of_newly_unblocked_tasks).
        """
        if status not in ("in_progress", "completed", "failed"):
            msg = f"Invalid status: {status}"
            raise ValueError(msg)
        async with self._lock:
            task = self._get_or_raise(task_id)
            task.status = status  # type: ignore[assignment]
            task.result = result
            task.updated_at = datetime.now(tz=UTC)
            unblocked: list[TeamTask] = []
            if status == "completed":
                unblocked = self._unblock_dependents(task_id)
            return task, unblocked

    async def list_tasks(
        self,
        status: str | None = None,
        assignee: str | None = None,
    ) -> list[TeamTask]:
        """List tasks with optional filters."""
        async with self._lock:
            tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if assignee is not None:
            tasks = [t for t in tasks if t.assignee == assignee]
        return tasks

    async def get_task(self, task_id: str) -> TeamTask | None:
        """Get a single task by ID."""
        async with self._lock:
            return self._tasks.get(task_id)

    async def get_available_tasks(self, for_teammate: str) -> list[TeamTask]:  # noqa: ARG002
        """Return unassigned pending tasks (all dependencies resolved)."""
        async with self._lock:
            return [t for t in self._tasks.values() if t.status == "pending" and t.assignee is None]

    def _get_or_raise(self, task_id: str) -> TeamTask:
        task = self._tasks.get(task_id)
        if task is None:
            msg = f"Task not found: {task_id}"
            raise KeyError(msg)
        return task

    def _validate_dependencies(self, blocked_by: list[str]) -> None:
        """Validate that all dependency task IDs exist."""
        for dep_id in blocked_by:
            if dep_id not in self._tasks:
                msg = f"Dependency task not found: {dep_id}"
                raise KeyError(msg)

    def _unblock_dependents(self, completed_task_id: str) -> list[TeamTask]:
        """Transition blocked tasks to pending if all their blockers are completed.

        Returns the list of tasks that were unblocked.
        """
        unblocked: list[TeamTask] = []
        for task in self._tasks.values():
            if task.status != "blocked":
                continue
            if completed_task_id not in task.blocked_by:
                continue
            all_done = all(
                self._tasks[dep_id].status == "completed"
                for dep_id in task.blocked_by
                if dep_id in self._tasks
            )
            if all_done:
                task.status = "pending"
                task.updated_at = datetime.now(tz=UTC)
                unblocked.append(task)
        return unblocked
