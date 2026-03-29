"""Plan context builder for sub-agent injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_sdk.memory.plan import Plan


def build_sub_agent_plan_context(
    plan: Plan,
    current_step_index: int | None = None,
) -> str:
    """Build a plan context summary for sub-agent system prompt injection.

    Generates a markdown summary with:
    - The parent plan's goal
    - All steps with their current status
    - Current step highlighted with → when current_step_index is provided

    Args:
        plan: The parent agent's plan.
        current_step_index: 0-based index of the step this sub-agent implements.

    Returns:
        Markdown string for injection into sub-agent system prompt.
        Empty string if plan has no goal.
    """
    if not plan.goal:
        return ""

    lines = ["## Parent Plan Context", f"Goal: {plan.goal}", ""]

    for i, step in enumerate(plan.steps):
        badge = "FAILED" if step.status == "failed" else step.status
        prefix = "→" if i == current_step_index else " "
        lines.append(f"{prefix} {i + 1}. [{badge}] {step.description}")

    if current_step_index is not None and 0 <= current_step_index < len(plan.steps):
        step_desc = plan.steps[current_step_index].description
        lines.append(f"\nYou are implementing step {current_step_index + 1}: {step_desc}")
        lines.append("Focus on this step. The parent agent handles the overall plan.")

    return "\n".join(lines)
