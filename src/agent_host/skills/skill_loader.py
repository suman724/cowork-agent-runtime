"""SkillLoader — loads skills from built-in, workspace YAML, and policy bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from agent_host.skills.models import SkillDefinition

logger = structlog.get_logger()

# Built-in skill definitions
_BUILTIN_SKILLS: list[SkillDefinition] = [
    SkillDefinition(
        name="search_codebase",
        description=(
            "Systematically search the codebase: "
            "grep for patterns, read matching files, report findings."
        ),
        system_prompt_additions=(
            "You are performing a focused codebase search. "
            "Use grep/find via RunCommand to locate relevant code, "
            "then ReadFile to examine matches. Report a clear summary."
        ),
        tool_subset=["ReadFile", "RunCommand"],
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
            },
            "required": ["query"],
        },
        max_steps=15,
    ),
    SkillDefinition(
        name="edit_and_verify",
        description="Edit a file, read it back to verify, and run tests.",
        system_prompt_additions=(
            "You are performing an edit-and-verify cycle. "
            "1. Write the file changes. "
            "2. Read the file back to verify correctness. "
            "3. Run relevant tests to confirm nothing broke."
        ),
        tool_subset=["ReadFile", "WriteFile", "RunCommand"],
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "changes": {"type": "string", "description": "Description of changes to make."},
            },
            "required": ["file_path", "changes"],
        },
        max_steps=15,
    ),
    SkillDefinition(
        name="debug_error",
        description="Reproduce an error, trace its root cause, and identify a fix.",
        system_prompt_additions=(
            "You are debugging an error. "
            "1. Reproduce the error by running the relevant command. "
            "2. Read relevant source files to trace the root cause. "
            "3. Identify the fix and report your findings."
        ),
        tool_subset=["ReadFile", "RunCommand"],
        input_schema={
            "type": "object",
            "properties": {
                "error_description": {"type": "string"},
                "reproduction_command": {"type": "string"},
            },
            "required": ["error_description"],
        },
        max_steps=15,
    ),
]


class SkillLoader:
    """Loads skill definitions from multiple sources.

    Sources (in priority order):
    1. Built-in skills (Python, always available)
    2. Workspace skills (YAML files in {workspace_dir}/.cowork/skills/)
    3. Policy bundle skills (from policyBundle.skills field)
    """

    def __init__(
        self,
        workspace_dir: str | None = None,
        policy_skills: list[dict[str, Any]] | None = None,
    ) -> None:
        self._workspace_dir = workspace_dir
        self._policy_skills = policy_skills or []

    def load_all(self) -> list[SkillDefinition]:
        """Load skills from all sources, deduplicating by name."""
        skills: dict[str, SkillDefinition] = {}

        # 1. Built-in skills
        for skill in _BUILTIN_SKILLS:
            skills[skill.name] = skill

        # 2. Workspace skills (override built-in if same name)
        for skill in self._load_workspace_skills():
            skills[skill.name] = skill

        # 3. Policy skills (override workspace if same name)
        for skill in self._load_policy_skills():
            skills[skill.name] = skill

        return list(skills.values())

    def _load_workspace_skills(self) -> list[SkillDefinition]:
        """Load skills from workspace YAML files."""
        if not self._workspace_dir:
            return []

        skills_dir = Path(self._workspace_dir) / ".cowork" / "skills"
        if not skills_dir.is_dir():
            return []

        skills: list[SkillDefinition] = []
        for yaml_file in sorted(skills_dir.glob("*.yaml")) + sorted(skills_dir.glob("*.yml")):
            try:
                skill = self._parse_yaml_skill(yaml_file)
                if skill:
                    skills.append(skill)
            except Exception:
                logger.warning(
                    "skill_load_failed",
                    path=str(yaml_file),
                    exc_info=True,
                )
        return skills

    def _load_policy_skills(self) -> list[SkillDefinition]:
        """Load skills from policy bundle."""
        skills: list[SkillDefinition] = []
        for skill_data in self._policy_skills:
            try:
                skill = SkillDefinition(
                    name=skill_data["name"],
                    description=skill_data.get("description", ""),
                    system_prompt_additions=skill_data.get("systemPromptAdditions", ""),
                    tool_subset=skill_data.get("toolSubset"),
                    input_schema=skill_data.get("inputSchema", {}),
                    max_steps=skill_data.get("maxSteps", 15),
                )
                skills.append(skill)
            except (KeyError, TypeError):
                logger.warning(
                    "policy_skill_invalid",
                    skill_data=skill_data,
                    exc_info=True,
                )
        return skills

    @staticmethod
    def _parse_yaml_skill(path: Path) -> SkillDefinition | None:
        """Parse a single YAML skill file."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("yaml_not_available", path=str(path))
            return None

        text = path.read_text()
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "name" not in data:
            return None

        return SkillDefinition(
            name=data["name"],
            description=data.get("description", ""),
            system_prompt_additions=data.get("system_prompt_additions", ""),
            tool_subset=data.get("tool_subset"),
            input_schema=data.get("input_schema", {}),
            examples=data.get("examples"),
            max_steps=data.get("max_steps", 15),
        )
