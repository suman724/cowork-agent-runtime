"""SkillLoader — loads skills from built-in markdown and user SKILL.md directories.

Sources (in priority order):
1. Built-in skills (embedded markdown strings, always available)
2. User skills (~/.cowork/skills/<name>/SKILL.md, directory-based markdown)

Policy bundle skills are accepted but not yet wired (Phase 3+).

Progressive disclosure:
- Stage 1 (metadata): Parse frontmatter only from each SKILL.md (~100 tokens per skill)
- Stage 2 (full content): Load SKILL.md body + supporting .md files on invocation
"""

from __future__ import annotations

import dataclasses
import platform
import re
from pathlib import Path
from typing import Any

import structlog
import yaml

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


def _split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Split '---\\nYAML\\n---\\nbody' into (dict, body).

    Returns (None, text) if no valid frontmatter found.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return None, text
    end = stripped.find("\n---", 3)
    if end == -1:
        return None, text
    fm_text = stripped[3:end]
    body = stripped[end + 4 :]  # skip \n---
    try:
        metadata = yaml.safe_load(fm_text)
        return (metadata, body) if isinstance(metadata, dict) else (None, text)
    except yaml.YAMLError:
        return None, text


def _dir_name_to_skill_name(dir_name: str) -> str:
    """Convert a directory name to a skill name (hyphens → underscores)."""
    return dir_name.replace("-", "_")


def _discover_skill_dirs(base_dir: Path) -> list[Path]:
    """Find skill directories: each must contain SKILL.md."""
    if not base_dir.is_dir():
        return []
    dirs = []
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir() and (entry / "SKILL.md").is_file():
            dirs.append(entry)
    return dirs


# ---------------------------------------------------------------------------
# Built-in skill definitions as embedded markdown strings
# ---------------------------------------------------------------------------

_BUILTIN_SKILLS_MD: dict[str, str] = {
    "search_codebase": """\
---
name: search_codebase
description: >-
  Systematically search the codebase: grep for patterns,
  read matching files, report findings.
tool_subset:
  - ReadFile
  - RunCommand
max_steps: 15
---

You are performing a focused codebase search.
Use grep/find via RunCommand to locate relevant code,
then ReadFile to examine matches. Report a clear summary.
""",
    "edit_and_verify": """\
---
name: edit_and_verify
description: Edit a file, read it back to verify, and run tests.
tool_subset:
  - ReadFile
  - WriteFile
  - RunCommand
max_steps: 15
---

You are performing an edit-and-verify cycle.
1. Write the file changes.
2. Read the file back to verify correctness.
3. Run relevant tests to confirm nothing broke.
""",
    "debug_error": """\
---
name: debug_error
description: Reproduce an error, trace its root cause, and identify a fix.
tool_subset:
  - ReadFile
  - RunCommand
max_steps: 15
---

You are debugging an error.
1. Reproduce the error by running the relevant command.
2. Read relevant source files to trace the root cause.
3. Identify the fix and report your findings.
""",
}


def _parse_skill_from_frontmatter(
    metadata: dict[str, Any],
    body: str,
    *,
    source_dir: str | None = None,
    populate_content: bool = False,
) -> SkillDefinition | None:
    """Build a SkillDefinition from parsed frontmatter metadata.

    Args:
        metadata: Parsed YAML frontmatter dict.
        body: Markdown body after the frontmatter.
        source_dir: Skill directory path (None for built-in).
        populate_content: If True, set prompt_content to body (for built-in skills).

    Returns:
        SkillDefinition or None if description is missing.
    """
    description = metadata.get("description")
    if not description:
        return None

    name = metadata.get("name", "")
    tool_subset = metadata.get("tool_subset")
    if tool_subset is not None and not isinstance(tool_subset, list):
        tool_subset = None

    return SkillDefinition(
        name=name,
        description=str(description).strip(),
        prompt_content=body.strip() if populate_content else "",
        source_dir=source_dir,
        tool_subset=tool_subset,
        input_schema=metadata.get("input_schema", {}),
        max_steps=int(metadata.get("max_steps", 15)),
        disable_model_invocation=bool(metadata.get("disable_model_invocation", False)),
        user_invocable=bool(metadata.get("user_invocable", True)),
    )


def _parse_skill_metadata(skill_dir: Path) -> SkillDefinition | None:
    """Stage 1: Parse SKILL.md frontmatter only.

    Returns SkillDefinition with empty prompt_content.
    """
    skill_md = skill_dir / "SKILL.md"
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        logger.warning("skill_md_read_failed", path=str(skill_md))
        return None

    metadata, _body = _split_frontmatter(text)
    if metadata is None:
        logger.warning("skill_invalid_frontmatter", path=str(skill_md))
        return None

    skill = _parse_skill_from_frontmatter(
        metadata, _body, source_dir=str(skill_dir), populate_content=False
    )
    if skill is None:
        logger.warning("skill_missing_description", path=str(skill_md))
        return None

    # Derive name from directory if not specified in frontmatter
    if not skill.name:
        derived_name = _dir_name_to_skill_name(skill_dir.name)
        skill = dataclasses.replace(skill, name=derived_name)

    return skill


def _resolve_script_dirs(skill_dir: Path, metadata: dict[str, Any] | None) -> list[Path]:
    """Resolve script directories from frontmatter ``scripts_dir``.

    Supports:
    - ``scripts_dir: scripts`` (single string, relative to skill dir)
    - ``scripts_dir: [scripts, scripts/office]`` (list of strings)

    Falls back to ``scripts/`` if no ``scripts_dir`` key and the directory exists.
    Path traversal is prevented by verifying resolved paths stay within skill_dir.
    """
    if not metadata or "scripts_dir" not in metadata:
        default = skill_dir / "scripts"
        return [default] if default.is_dir() else []

    raw = metadata["scripts_dir"]
    if isinstance(raw, str):
        dirs = [raw]
    elif isinstance(raw, list):
        dirs = [str(d) for d in raw]
    else:
        return []

    resolved: list[Path] = []
    skill_root = skill_dir.resolve()
    for d in dirs:
        p = (skill_dir / d).resolve()
        if p.is_dir() and p.is_relative_to(skill_root):
            resolved.append(p)
        else:
            logger.warning("scripts_dir_invalid", path=d, skill=str(skill_dir))
    return resolved


def _collect_scripts(script_dirs: list[Path]) -> list[Path]:
    """Collect non-hidden direct children (files only) from each script directory.

    Non-recursive: only lists immediate files in each directory.  If a skill needs
    nested scripts listed, the author declares multiple directories in ``scripts_dir``.
    """
    seen: set[Path] = set()
    scripts: list[Path] = []
    for sdir in script_dirs:
        for f in sorted(sdir.iterdir()):
            if f.is_file() and f not in seen and not f.name.startswith("."):
                seen.add(f)
                scripts.append(f)
    return scripts


class SkillLoader:
    """Loads skill definitions from multiple sources with progressive disclosure.

    Sources (in priority order):
    1. Built-in skills (embedded markdown, always available, prompt_content pre-populated)
    2. User skills (directory-based SKILL.md, prompt_content lazy-loaded)
    3. Policy bundle skills (Phase 3+, dict-based)
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

        # 1. Built-in skills (prompt_content pre-populated)
        for skill in self._load_builtin_skills():
            skills[skill.name] = skill

        # 2. User skills (override built-in if same name)
        for skill in self._load_user_skills():
            skills[skill.name] = skill

        # 3. Policy skills — Phase 3+ (override user if same name)
        for skill in self._load_policy_skills():
            skills[skill.name] = skill

        return list(skills.values())

    @staticmethod
    def _load_builtin_skills() -> list[SkillDefinition]:
        """Load built-in skills from embedded markdown strings."""
        skills: list[SkillDefinition] = []
        for _name, md_text in _BUILTIN_SKILLS_MD.items():
            metadata, body = _split_frontmatter(md_text)
            if metadata is None:
                continue
            skill = _parse_skill_from_frontmatter(
                metadata, body, source_dir=None, populate_content=True
            )
            if skill is not None:
                skills.append(skill)
        return skills

    def _load_user_skills(self) -> list[SkillDefinition]:
        """Load skills from user directories (~/.cowork/skills/<name>/SKILL.md)."""
        skill_dirs = _discover_skill_dirs(self._user_skills_dir)
        skills: list[SkillDefinition] = []
        for skill_dir in skill_dirs:
            try:
                skill = _parse_skill_metadata(skill_dir)
                if skill:
                    skills.append(skill)
                    logger.info(
                        "user_skill_loaded",
                        name=skill.name,
                        path=str(skill_dir),
                    )
            except Exception:
                logger.warning(
                    "skill_load_failed",
                    path=str(skill_dir),
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
                    prompt_content=skill_data.get("promptContent", ""),
                    tool_subset=skill_data.get("toolSubset"),
                    input_schema=skill_data.get("inputSchema", {}),
                    max_steps=skill_data.get("maxSteps", 15),
                    disable_model_invocation=skill_data.get("disableModelInvocation", False),
                    user_invocable=skill_data.get("userInvocable", True),
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
    def load_skill_content(skill: SkillDefinition) -> SkillDefinition:
        """Stage 2: Load full SKILL.md body + supporting .md files.

        If prompt_content is already populated (built-in or previously loaded),
        returns the skill as-is. Otherwise reads source_dir/SKILL.md body and
        appends all supporting .md files (sorted, with section headers).

        Returns a new frozen SkillDefinition with prompt_content populated.
        """
        if skill.prompt_content or skill.source_dir is None:
            return skill

        skill_dir = Path(skill.source_dir)
        skill_md = skill_dir / "SKILL.md"

        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            logger.warning("skill_content_read_failed", path=str(skill_md))
            return skill

        metadata, body = _split_frontmatter(text)
        full_content = body.strip()

        # Collect supporting .md files (everything except SKILL.md)
        supporting_parts: list[str] = []
        for md_file in sorted(skill_dir.glob("*.md")):
            if md_file.name == "SKILL.md":
                continue
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if content:
                    section_name = md_file.stem.replace("-", " ").replace("_", " ").title()
                    supporting_parts.append(f"## {section_name}\n\n{content}")
            except OSError:
                logger.warning("supporting_file_read_failed", path=str(md_file))

        # Collect scripts from frontmatter scripts_dir or default scripts/ fallback
        script_dirs = _resolve_script_dirs(skill_dir, metadata)
        script_files = _collect_scripts(script_dirs)
        if script_files:
            script_lines = [f"- `{f.relative_to(skill_dir)}`: `{f}`" for f in script_files]
            supporting_parts.append(
                "## Available Scripts\n\n"
                f"Skill directory: `{skill_dir}`\n\n"
                + "\n".join(script_lines)
                + "\n\nRun scripts using their full absolute paths shown above."
            )

        if supporting_parts:
            full_content += "\n\n---\n\n" + "\n\n".join(supporting_parts)

        return dataclasses.replace(skill, prompt_content=full_content)


def substitute_arguments(content: str, arguments: dict[str, Any]) -> str:
    """Replace $ARGUMENTS and $ARGUMENTS[N] placeholders in skill content.

    - $ARGUMENTS → all argument values joined with space
    - $ARGUMENTS[N] → positional argument by 0-based index
    - Out-of-range index → empty string
    - No arguments → placeholders replaced with empty strings
    """
    if not content:
        return content

    arg_values = list(arguments.values())
    args_string = " ".join(str(v) for v in arg_values)

    # Replace indexed $ARGUMENTS[N] first (more specific pattern)
    def _replace_indexed(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return str(arg_values[idx]) if idx < len(arg_values) else ""

    result = re.sub(r"\$ARGUMENTS\[(\d+)\]", _replace_indexed, content)
    result = result.replace("$ARGUMENTS", args_string)
    return result
