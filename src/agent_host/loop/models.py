"""Data models for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LoopResult:
    """Result of an agent loop execution."""

    reason: str  # "completed" | "cancelled" | "max_steps_exceeded" | "error"
    text: str = ""
    step_count: int = 0


@dataclass(frozen=True)
class ToolCallResult:
    """Result of executing a single tool call."""

    tool_call_id: str
    tool_name: str
    status: str
    result_text: str
    arguments: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Any] | None = None
