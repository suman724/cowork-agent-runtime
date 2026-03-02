"""Tests for TaskTracker — CRUD and rendering."""

from __future__ import annotations

from agent_host.memory.task_tracker import TaskTracker


class TestTaskTrackerCRUD:
    def test_create_task(self) -> None:
        tracker = TaskTracker()
        task_id = tracker.create_task("Implement auth endpoints")
        assert task_id.startswith("t-")
        assert len(tracker.tasks) == 1
        assert tracker.tasks[0].content == "Implement auth endpoints"
        assert tracker.tasks[0].status == "pending"

    def test_update_task_status(self) -> None:
        tracker = TaskTracker()
        task_id = tracker.create_task("Write tests")
        assert tracker.update_task(task_id, status="in_progress")
        assert tracker.get_task(task_id) is not None
        assert tracker.get_task(task_id).status == "in_progress"  # type: ignore[union-attr]

    def test_update_task_content(self) -> None:
        tracker = TaskTracker()
        task_id = tracker.create_task("Old task")
        assert tracker.update_task(task_id, content="Updated task")
        assert tracker.get_task(task_id).content == "Updated task"  # type: ignore[union-attr]

    def test_update_nonexistent_task(self) -> None:
        tracker = TaskTracker()
        assert not tracker.update_task("bogus-id", status="completed")

    def test_get_nonexistent_task(self) -> None:
        tracker = TaskTracker()
        assert tracker.get_task("bogus-id") is None

    def test_multiple_tasks(self) -> None:
        tracker = TaskTracker()
        id1 = tracker.create_task("First")
        id2 = tracker.create_task("Second")
        id3 = tracker.create_task("Third")
        assert len(tracker.tasks) == 3
        assert id1 != id2 != id3


class TestTaskTrackerRender:
    def test_render_empty(self) -> None:
        tracker = TaskTracker()
        assert tracker.render() == ""

    def test_render_with_tasks(self) -> None:
        tracker = TaskTracker()
        tracker.create_task("Implement auth")
        tracker.create_task("Write tests")
        rendered = tracker.render()
        assert "## Current Tasks" in rendered
        assert "[pending] Implement auth" in rendered
        assert "[pending] Write tests" in rendered

    def test_render_with_mixed_statuses(self) -> None:
        tracker = TaskTracker()
        id1 = tracker.create_task("Done task")
        tracker.create_task("Pending task")
        tracker.update_task(id1, status="completed")
        rendered = tracker.render()
        assert "[completed] Done task" in rendered
        assert "[pending] Pending task" in rendered


class TestTaskTrackerCheckpoint:
    def test_round_trip(self) -> None:
        tracker = TaskTracker()
        id1 = tracker.create_task("Task A")
        tracker.create_task("Task B")
        tracker.update_task(id1, status="completed")

        data = tracker.to_checkpoint()
        restored = TaskTracker.from_checkpoint(data)

        assert len(restored.tasks) == 2
        assert restored.get_task(id1) is not None
        assert restored.get_task(id1).status == "completed"  # type: ignore[union-attr]

    def test_empty_checkpoint(self) -> None:
        restored = TaskTracker.from_checkpoint([])
        assert len(restored.tasks) == 0
