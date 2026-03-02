"""Tests for WorkingMemory — structured agent state."""

from __future__ import annotations

from agent_host.memory.plan import Plan, PlanStep
from agent_host.memory.working_memory import WorkingMemory


class TestWorkingMemoryRender:
    def test_render_empty(self) -> None:
        wm = WorkingMemory()
        assert wm.render() == ""

    def test_render_with_tasks_only(self) -> None:
        wm = WorkingMemory()
        wm.task_tracker.create_task("Implement feature")
        rendered = wm.render()
        assert "## Current Tasks" in rendered
        assert "Implement feature" in rendered

    def test_render_with_plan_only(self) -> None:
        wm = WorkingMemory()
        wm.plan = Plan(
            goal="Add auth",
            steps=[PlanStep(description="Create schema")],
        )
        rendered = wm.render()
        assert "## Current Plan" in rendered
        assert "Goal: Add auth" in rendered

    def test_render_with_notes_only(self) -> None:
        wm = WorkingMemory()
        wm.notes.append("User prefers JWT over sessions")
        rendered = wm.render()
        assert "## Notes" in rendered
        assert "User prefers JWT" in rendered

    def test_render_all_sections(self) -> None:
        wm = WorkingMemory()
        wm.task_tracker.create_task("Implementing auth endpoints")
        wm.plan = Plan(goal="Add authentication", steps=[PlanStep(description="Step 1")])
        wm.notes.append("Important note")

        rendered = wm.render()
        assert "## Current Tasks" in rendered
        assert "## Current Plan" in rendered
        assert "## Notes" in rendered


class TestWorkingMemoryCheckpoint:
    def test_round_trip(self) -> None:
        wm = WorkingMemory()
        tid = wm.task_tracker.create_task("Task A")
        wm.task_tracker.update_task(tid, status="in_progress")
        wm.plan = Plan(
            goal="Build feature",
            steps=[PlanStep(description="First step", status="completed")],
        )
        wm.notes.append("Remember: user prefers dark mode")
        wm.notes.append("API uses v2 endpoints")

        data = wm.to_checkpoint()
        restored = WorkingMemory.from_checkpoint(data)

        # Tasks preserved
        assert len(restored.task_tracker.tasks) == 1
        assert restored.task_tracker.tasks[0].status == "in_progress"

        # Plan preserved
        assert restored.plan is not None
        assert restored.plan.goal == "Build feature"
        assert len(restored.plan.steps) == 1
        assert restored.plan.steps[0].status == "completed"

        # Notes preserved
        assert len(restored.notes) == 2
        assert "dark mode" in restored.notes[0]

    def test_empty_checkpoint(self) -> None:
        restored = WorkingMemory.from_checkpoint({})
        assert len(restored.task_tracker.tasks) == 0
        assert restored.plan is None
        assert len(restored.notes) == 0

    def test_partial_checkpoint(self) -> None:
        """Checkpoint with only some fields populated."""
        data = {
            "tasks": [],
            "plan": None,
            "notes": ["just a note"],
        }
        restored = WorkingMemory.from_checkpoint(data)
        assert len(restored.notes) == 1
        assert restored.plan is None
