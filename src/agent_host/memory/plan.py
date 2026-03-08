"""Plan — structured agent plan for multi-step work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PlanStep:
    """A single step in an agent plan."""

    description: str
    status: Literal["pending", "in_progress", "completed", "skipped"] = "pending"


@dataclass
class Plan:
    """A structured plan the agent creates and follows."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    def render(self) -> str:
        """Render plan as text for injection into system prompt."""
        if not self.goal:
            return ""
        lines = [f"## Current Plan\nGoal: {self.goal}"]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"{i}. [{step.status}] {step.description}")
        in_progress = [(i, s) for i, s in enumerate(self.steps) if s.status == "in_progress"]
        has_pending = any(s.status == "pending" for s in self.steps)
        if in_progress:
            indices = ", ".join(str(i) for i, _ in in_progress)
            lines.append(
                f"\nIMPORTANT: Step(s) {indices} are still marked in_progress. "
                "When you finish a step, you MUST call "
                "UpdatePlanStep(stepIndex, 'completed') before moving to the next step."
            )
        elif has_pending:
            lines.append(
                "\nCall UpdatePlanStep(stepIndex, 'in_progress') when you start a step, "
                "then UpdatePlanStep(stepIndex, 'completed') when you finish it."
            )
        return "\n".join(lines)

    def to_checkpoint(self) -> dict[str, Any]:
        """Serialize for checkpoint persistence."""
        return {
            "goal": self.goal,
            "steps": [{"description": s.description, "status": s.status} for s in self.steps],
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> Plan:
        """Restore from checkpoint data."""
        steps = [
            PlanStep(
                description=s["description"],
                status=s.get("status", "pending"),
            )
            for s in data.get("steps", [])
        ]
        return cls(goal=data.get("goal", ""), steps=steps)
