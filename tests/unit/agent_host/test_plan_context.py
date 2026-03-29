"""Tests for plan context builder — sub-agent plan injection."""

from __future__ import annotations

from agent_sdk.memory.plan import Plan, PlanStep

from agent_host.loop.plan_context import build_sub_agent_plan_context


class TestBuildSubAgentPlanContext:
    def test_with_current_step(self) -> None:
        plan = Plan(
            goal="Build auth system",
            steps=[
                PlanStep(description="Create DB schema", status="completed"),
                PlanStep(description="Implement endpoints", status="in_progress"),
                PlanStep(description="Write tests", status="pending"),
            ],
        )
        ctx = build_sub_agent_plan_context(plan, current_step_index=1)
        assert "Build auth system" in ctx
        assert "→" in ctx
        assert "You are implementing step 2: Implement endpoints" in ctx

    def test_shows_all_statuses(self) -> None:
        plan = Plan(
            goal="Deploy",
            steps=[
                PlanStep(description="Build", status="completed"),
                PlanStep(description="Push", status="failed"),
                PlanStep(description="Deploy", status="in_progress"),
                PlanStep(description="Verify", status="pending"),
                PlanStep(description="Old", status="skipped"),
            ],
        )
        ctx = build_sub_agent_plan_context(plan, current_step_index=2)
        assert "[completed]" in ctx
        assert "[FAILED]" in ctx
        assert "[in_progress]" in ctx
        assert "[pending]" in ctx
        assert "[skipped]" in ctx

    def test_no_step_index(self) -> None:
        plan = Plan(goal="Test", steps=[PlanStep(description="Step 1")])
        ctx = build_sub_agent_plan_context(plan, current_step_index=None)
        assert "Test" in ctx
        assert "Step 1" in ctx
        assert "You are implementing" not in ctx

    def test_no_plan_goal(self) -> None:
        plan = Plan(goal="", steps=[PlanStep(description="X")])
        ctx = build_sub_agent_plan_context(plan)
        assert ctx == ""

    def test_step_numbering_is_1_based(self) -> None:
        plan = Plan(
            goal="Test",
            steps=[PlanStep(description="First"), PlanStep(description="Second")],
        )
        ctx = build_sub_agent_plan_context(plan, current_step_index=0)
        assert "1. [pending] First" in ctx
        assert "2. [pending] Second" in ctx
        assert "You are implementing step 1: First" in ctx

    def test_current_step_highlighted_others_not(self) -> None:
        plan = Plan(
            goal="Test",
            steps=[PlanStep(description="A"), PlanStep(description="B")],
        )
        ctx = build_sub_agent_plan_context(plan, current_step_index=1)
        lines = ctx.split("\n")
        step_lines = [line for line in lines if "[pending]" in line]
        assert len(step_lines) == 2
        # First step should have space prefix, second should have → prefix
        assert any("→" in line and "B" in line for line in step_lines)
        assert any(line.startswith(" ") and "A" in line for line in step_lines)
