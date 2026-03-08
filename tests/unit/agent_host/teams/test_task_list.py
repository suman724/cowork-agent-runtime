"""Tests for SharedTaskList."""

from __future__ import annotations

import asyncio

import pytest

from agent_host.teams.task_list import SharedTaskList


class TestCreateTask:
    async def test_create_returns_task_with_id(self) -> None:
        tl = SharedTaskList()
        task = await tl.create_task("Build API", "Implement REST endpoints")
        assert len(task.task_id) == 12
        assert task.title == "Build API"
        assert task.status == "pending"

    async def test_create_with_creator(self) -> None:
        tl = SharedTaskList()
        task = await tl.create_task("Test", "Run tests", created_by="lead")
        assert task.created_by == "lead"

    async def test_create_blocked_task(self) -> None:
        tl = SharedTaskList()
        t1 = await tl.create_task("First", "Do first")
        t2 = await tl.create_task("Second", "Do second", blocked_by=[t1.task_id])
        assert t2.status == "blocked"
        assert t2.blocked_by == [t1.task_id]

    async def test_create_blocked_by_nonexistent_raises(self) -> None:
        tl = SharedTaskList()
        with pytest.raises(KeyError, match="Dependency task not found"):
            await tl.create_task("Bad", "Bad dep", blocked_by=["nonexistent"])


class TestAssignTask:
    async def test_assign_sets_in_progress(self) -> None:
        tl = SharedTaskList()
        task = await tl.create_task("Work", "Do work")
        assigned = await tl.assign_task(task.task_id, "worker-1")
        assert assigned.status == "in_progress"
        assert assigned.assignee == "worker-1"

    async def test_assign_nonexistent_raises(self) -> None:
        tl = SharedTaskList()
        with pytest.raises(KeyError, match="Task not found"):
            await tl.assign_task("nope", "worker")

    async def test_assign_non_pending_raises(self) -> None:
        tl = SharedTaskList()
        task = await tl.create_task("Work", "Do work")
        await tl.assign_task(task.task_id, "worker-1")
        with pytest.raises(ValueError, match="status is 'in_progress'"):
            await tl.assign_task(task.task_id, "worker-2")


class TestUpdateStatus:
    async def test_complete_task(self) -> None:
        tl = SharedTaskList()
        task = await tl.create_task("Work", "Do work")
        await tl.assign_task(task.task_id, "w")
        updated = await tl.update_status(task.task_id, "completed", result="Done successfully")
        assert updated.status == "completed"
        assert updated.result == "Done successfully"

    async def test_fail_task(self) -> None:
        tl = SharedTaskList()
        task = await tl.create_task("Work", "Do work")
        updated = await tl.update_status(task.task_id, "failed")
        assert updated.status == "failed"

    async def test_invalid_status_raises(self) -> None:
        tl = SharedTaskList()
        task = await tl.create_task("Work", "Do work")
        with pytest.raises(ValueError, match="Invalid status"):
            await tl.update_status(task.task_id, "bogus")


class TestDependencyResolution:
    async def test_completing_blocker_unblocks_dependent(self) -> None:
        tl = SharedTaskList()
        t1 = await tl.create_task("First", "Do first")
        t2 = await tl.create_task("Second", "Do second", blocked_by=[t1.task_id])
        assert t2.status == "blocked"

        await tl.update_status(t1.task_id, "completed")
        refreshed = await tl.get_task(t2.task_id)
        assert refreshed is not None
        assert refreshed.status == "pending"

    async def test_partial_unblock_stays_blocked(self) -> None:
        tl = SharedTaskList()
        t1 = await tl.create_task("A", "a")
        t2 = await tl.create_task("B", "b")
        t3 = await tl.create_task("C", "c", blocked_by=[t1.task_id, t2.task_id])
        assert t3.status == "blocked"

        await tl.update_status(t1.task_id, "completed")
        refreshed = await tl.get_task(t3.task_id)
        assert refreshed is not None
        assert refreshed.status == "blocked"

        await tl.update_status(t2.task_id, "completed")
        refreshed = await tl.get_task(t3.task_id)
        assert refreshed is not None
        assert refreshed.status == "pending"


class TestListAndFilter:
    async def test_list_all(self) -> None:
        tl = SharedTaskList()
        await tl.create_task("A", "a")
        await tl.create_task("B", "b")
        tasks = await tl.list_tasks()
        assert len(tasks) == 2

    async def test_filter_by_status(self) -> None:
        tl = SharedTaskList()
        t1 = await tl.create_task("A", "a")
        await tl.create_task("B", "b")
        await tl.assign_task(t1.task_id, "w")
        pending = await tl.list_tasks(status="pending")
        assert len(pending) == 1

    async def test_filter_by_assignee(self) -> None:
        tl = SharedTaskList()
        t1 = await tl.create_task("A", "a")
        await tl.create_task("B", "b")
        await tl.assign_task(t1.task_id, "alice")
        assigned = await tl.list_tasks(assignee="alice")
        assert len(assigned) == 1
        assert assigned[0].assignee == "alice"

    async def test_get_available_tasks(self) -> None:
        tl = SharedTaskList()
        t1 = await tl.create_task("A", "a")
        t2 = await tl.create_task("B", "b")
        await tl.create_task("C", "c", blocked_by=[t1.task_id])
        await tl.assign_task(t2.task_id, "w")

        available = await tl.get_available_tasks("worker")
        assert len(available) == 1
        assert available[0].task_id == t1.task_id


class TestConcurrency:
    async def test_concurrent_creates(self) -> None:
        tl = SharedTaskList()

        async def create_one(i: int) -> None:
            await tl.create_task(f"Task {i}", f"Description {i}")

        await asyncio.gather(*(create_one(i) for i in range(20)))
        tasks = await tl.list_tasks()
        assert len(tasks) == 20
