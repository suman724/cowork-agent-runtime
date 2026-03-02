"""Tests for Plan — structured agent plan."""

from __future__ import annotations

from agent_host.memory.plan import Plan, PlanStep


class TestPlanRender:
    def test_render_empty_goal(self) -> None:
        plan = Plan(goal="")
        assert plan.render() == ""

    def test_render_with_steps(self) -> None:
        plan = Plan(
            goal="Add user authentication",
            steps=[
                PlanStep(description="Create DB schema", status="completed"),
                PlanStep(description="Implement endpoints", status="in_progress"),
                PlanStep(description="Write tests", status="pending"),
            ],
        )
        rendered = plan.render()
        assert "## Current Plan" in rendered
        assert "Goal: Add user authentication" in rendered
        assert "1. [completed] Create DB schema" in rendered
        assert "2. [in_progress] Implement endpoints" in rendered
        assert "3. [pending] Write tests" in rendered

    def test_render_no_steps(self) -> None:
        plan = Plan(goal="Simple goal")
        rendered = plan.render()
        assert "Goal: Simple goal" in rendered


class TestPlanCheckpoint:
    def test_round_trip(self) -> None:
        plan = Plan(
            goal="Build feature",
            steps=[
                PlanStep(description="Step 1", status="completed"),
                PlanStep(description="Step 2", status="pending"),
            ],
        )
        data = plan.to_checkpoint()
        restored = Plan.from_checkpoint(data)

        assert restored.goal == "Build feature"
        assert len(restored.steps) == 2
        assert restored.steps[0].description == "Step 1"
        assert restored.steps[0].status == "completed"
        assert restored.steps[1].status == "pending"

    def test_empty_checkpoint(self) -> None:
        restored = Plan.from_checkpoint({})
        assert restored.goal == ""
        assert len(restored.steps) == 0
