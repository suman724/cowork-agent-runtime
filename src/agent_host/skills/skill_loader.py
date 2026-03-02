"""SkillLoader — loads skills from built-in Python and user YAML files.

Sources (in priority order):
1. Built-in skills (Python, always available)
2. User skills (~/.cowork/skills/*.yaml, user-level customization)

Policy bundle skills are accepted but not yet wired (Phase 3+).
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import structlog

from agent_host.skills.models import SkillDefinition

logger = structlog.get_logger()


def _default_user_skills_dir() -> Path:
    """Return the platform-specific user skills directory."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / ".cowork" / "skills"
    elif system == "Windows":
        app_data = Path.home() / "AppData" / "Roaming"
        return app_data / "cowork" / "skills"
    else:  # Linux and others
        return Path.home() / ".cowork" / "skills"


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
    2. User skills (YAML files in ~/.cowork/skills/)
    3. Policy bundle skills (Phase 3+, not yet wired)
    """

    def __init__(
        self,
        user_skills_dir: str | None = None,
        policy_skills: list[dict[str, Any]] | None = None,
    ) -> None:
        if user_skills_dir is not None:
            self._user_skills_dir = Path(user_skills_dir)
        else:
            self._user_skills_dir = _default_user_skills_dir()
        self._policy_skills = policy_skills or []

    def load_all(self) -> list[SkillDefinition]:
        """Load skills from all sources, deduplicating by name."""
        skills: dict[str, SkillDefinition] = {}

        # 1. Built-in skills
        for skill in _BUILTIN_SKILLS:
            skills[skill.name] = skill

        # 2. User skills (override built-in if same name)
        for skill in self._load_user_skills():
            skills[skill.name] = skill

        # 3. Policy skills — Phase 3+ (override user if same name)
        for skill in self._load_policy_skills():
            skills[skill.name] = skill

        return list(skills.values())

    def _load_user_skills(self) -> list[SkillDefinition]:
        """Load skills from user YAML files (~/.cowork/skills/)."""
        if not self._user_skills_dir.is_dir():
            return []

        skills: list[SkillDefinition] = []
        for yaml_file in sorted(self._user_skills_dir.glob("*.yaml")) + sorted(
            self._user_skills_dir.glob("*.yml")
        ):
            try:
                skill = self._parse_yaml_skill(yaml_file)
                if skill:
                    skills.append(skill)
                    logger.info(
                        "user_skill_loaded",
                        name=skill.name,
                        path=str(yaml_file),
                    )
            except Exception:
                logger.warning(
                    "skill_load_failed",
                    path=str(yaml_file),
                    exc_info=True,
                )
        return skills

    def _load_policy_skills(self) -> list[SkillDefinition]:
        """Load skills from policy bundle (Phase 3+)."""
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
