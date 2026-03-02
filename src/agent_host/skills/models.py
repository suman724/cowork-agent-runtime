"""Skill data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillDefinition:
    """A formalized multi-step workflow the agent can invoke."""

    name: str
    description: str
    system_prompt_additions: str = ""
    tool_subset: list[str] | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] | None = None
    max_steps: int = 15
