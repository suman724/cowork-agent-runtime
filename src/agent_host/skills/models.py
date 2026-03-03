"""Skill data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillDefinition:
    """A formalized multi-step workflow the agent can invoke.

    Skills use directory-based Markdown format with progressive disclosure:
    - Stage 1 (metadata): frontmatter only — name, description, flags (~100 tokens)
    - Stage 2 (full content): SKILL.md body + supporting .md files loaded on invocation

    Built-in skills have prompt_content pre-populated (no lazy loading).
    File-based skills have prompt_content="" until load_skill_content() is called.
    """

    name: str
    description: str
    prompt_content: str = ""
    source_dir: str | None = None
    tool_subset: list[str] | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 15
    disable_model_invocation: bool = False
    user_invocable: bool = True
