"""WorkingMemory — structured agent state injected into every LLM call."""

from __future__ import annotations

from typing import Any

from agent_host.memory.plan import Plan
from agent_host.memory.task_tracker import TaskTracker


class WorkingMemory:
    """Maintains structured agent state (tasks, plan, notes).

    Rendered as a text block and injected after the system prompt
    in every LLM call, preventing goal drift during long tasks.
    """

    def __init__(self) -> None:
        self.task_tracker = TaskTracker()
        self.plan: Plan | None = None
        self.notes: list[str] = []

    def render(self) -> str:
        """Render all working memory sections as a single text block."""
        parts: list[str] = []

        # Task list
        task_text = self.task_tracker.render()
        if task_text:
            parts.append(task_text)

        # Plan
        if self.plan:
            plan_text = self.plan.render()
            if plan_text:
                parts.append(plan_text)

        # Notes
        if self.notes:
            note_lines = ["## Notes"]
            for note in self.notes:
                note_lines.append(f"- {note}")
            parts.append("\n".join(note_lines))

        return "\n\n".join(parts)

    def to_checkpoint(self) -> dict[str, Any]:
        """Serialize for checkpoint persistence."""
        return {
            "tasks": self.task_tracker.to_checkpoint(),
            "plan": self.plan.to_checkpoint() if self.plan else None,
            "notes": list(self.notes),
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> WorkingMemory:
        """Restore from checkpoint data."""
        wm = cls()
        if data.get("tasks"):
            wm.task_tracker = TaskTracker.from_checkpoint(data["tasks"])
        if data.get("plan"):
            wm.plan = Plan.from_checkpoint(data["plan"])
        if data.get("notes"):
            wm.notes = list(data["notes"])
        return wm
