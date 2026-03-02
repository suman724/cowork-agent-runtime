"""Tests for SkillLoader — built-in and user YAML skill loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_host.skills.models import SkillDefinition
from agent_host.skills.skill_loader import SkillLoader


class TestBuiltinSkills:
    def test_load_builtin_skills(self) -> None:
        """Built-in skills should always load."""
        loader = SkillLoader()
        skills = loader.load_all()

        names = {s.name for s in skills}
        assert "search_codebase" in names
        assert "edit_and_verify" in names
        assert "debug_error" in names

    def test_builtin_skill_properties(self) -> None:
        """Built-in skills should have correct properties."""
        loader = SkillLoader()
        skills = {s.name: s for s in loader.load_all()}

        search = skills["search_codebase"]
        assert search.max_steps == 15
        assert search.tool_subset == ["ReadFile", "RunCommand"]
        assert "query" in search.input_schema.get("properties", {})

        edit = skills["edit_and_verify"]
        assert edit.tool_subset == ["ReadFile", "WriteFile", "RunCommand"]
        assert "file_path" in edit.input_schema.get("properties", {})

        debug = skills["debug_error"]
        assert debug.tool_subset == ["ReadFile", "RunCommand"]
        assert "error_description" in debug.input_schema.get("properties", {})


class TestUserSkills:
    def test_load_user_yaml(self, tmp_path: Path) -> None:
        """Should load skills from user YAML files."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        yaml_content = """\
name: custom_deploy
description: Deploy the application to staging.
system_prompt_additions: You are deploying to staging.
tool_subset:
  - RunCommand
input_schema:
  type: object
  properties:
    target:
      type: string
  required:
    - target
max_steps: 20
"""
        (skills_dir / "deploy.yaml").write_text(yaml_content)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = loader.load_all()
        by_name = {s.name: s for s in skills}

        assert "custom_deploy" in by_name
        deploy = by_name["custom_deploy"]
        assert deploy.description == "Deploy the application to staging."
        assert deploy.tool_subset == ["RunCommand"]
        assert deploy.max_steps == 20
        assert deploy.system_prompt_additions == "You are deploying to staging."

    def test_user_overrides_builtin(self, tmp_path: Path) -> None:
        """User skill with same name should override built-in."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        yaml_content = """\
name: search_codebase
description: Custom search override.
max_steps: 25
"""
        (skills_dir / "search.yaml").write_text(yaml_content)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        assert skills["search_codebase"].description == "Custom search override."
        assert skills["search_codebase"].max_steps == 25

    def test_no_user_skills_dir(self, tmp_path: Path) -> None:
        """Non-existent user skills dir should return only built-in skills."""
        loader = SkillLoader(user_skills_dir=str(tmp_path / "nonexistent"))
        skills = loader.load_all()
        assert len(skills) == 3  # only built-ins

    def test_invalid_yaml_skipped(self, tmp_path: Path) -> None:
        """Invalid YAML files should be skipped without crashing."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        # Invalid YAML (no name field)
        (skills_dir / "bad.yaml").write_text("description: no name field\n")

        # Valid YAML
        yaml_content = """\
name: valid_skill
description: A valid skill.
"""
        (skills_dir / "good.yaml").write_text(yaml_content)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = loader.load_all()
        names = {s.name for s in skills}
        assert "valid_skill" in names

    def test_yml_extension(self, tmp_path: Path) -> None:
        """Should load .yml files as well as .yaml."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        yaml_content = """\
name: yml_skill
description: Loaded from .yml file.
"""
        (skills_dir / "test.yml").write_text(yaml_content)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}
        assert "yml_skill" in skills


class TestPolicySkills:
    def test_load_policy_skills(self) -> None:
        """Should load skills from policy bundle data."""
        policy_skills: list[dict[str, Any]] = [
            {
                "name": "policy_scan",
                "description": "Run a security scan.",
                "systemPromptAdditions": "You are a security scanner.",
                "toolSubset": ["RunCommand"],
                "inputSchema": {
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                },
                "maxSteps": 10,
            }
        ]

        loader = SkillLoader(policy_skills=policy_skills)
        skills = {s.name: s for s in loader.load_all()}

        assert "policy_scan" in skills
        scan = skills["policy_scan"]
        assert scan.description == "Run a security scan."
        assert scan.system_prompt_additions == "You are a security scanner."
        assert scan.tool_subset == ["RunCommand"]
        assert scan.max_steps == 10

    def test_policy_overrides_user(self, tmp_path: Path) -> None:
        """Policy skill should override user skill with same name."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)

        yaml_content = """\
name: shared_skill
description: From user.
max_steps: 10
"""
        (skills_dir / "shared.yaml").write_text(yaml_content)

        policy_skills: list[dict[str, Any]] = [
            {
                "name": "shared_skill",
                "description": "From policy.",
                "maxSteps": 20,
            }
        ]

        loader = SkillLoader(user_skills_dir=str(skills_dir), policy_skills=policy_skills)
        skills = {s.name: s for s in loader.load_all()}

        assert skills["shared_skill"].description == "From policy."
        assert skills["shared_skill"].max_steps == 20

    def test_invalid_policy_skill_skipped(self) -> None:
        """Invalid policy skill data should be skipped."""
        policy_skills: list[dict[str, Any]] = [
            {"invalid": "no name field"},
            {"name": "valid_policy_skill", "description": "Works fine."},
        ]

        loader = SkillLoader(policy_skills=policy_skills)
        skills = {s.name: s for s in loader.load_all()}
        assert "valid_policy_skill" in skills

    def test_policy_skill_defaults(self) -> None:
        """Policy skill with minimal fields should use defaults."""
        policy_skills: list[dict[str, Any]] = [
            {"name": "minimal_skill"},
        ]

        loader = SkillLoader(policy_skills=policy_skills)
        skills = {s.name: s for s in loader.load_all()}

        minimal = skills["minimal_skill"]
        assert minimal.description == ""
        assert minimal.system_prompt_additions == ""
        assert minimal.tool_subset is None
        assert minimal.max_steps == 15


class TestSkillDefinitionModel:
    def test_frozen(self) -> None:
        """SkillDefinition should be immutable."""
        skill = SkillDefinition(name="test", description="test skill")
        try:
            skill.name = "changed"  # type: ignore[misc]
            msg = "Should have raised FrozenInstanceError"
            raise AssertionError(msg)
        except AttributeError:
            pass  # Expected — frozen dataclass

    def test_defaults(self) -> None:
        """SkillDefinition should have sensible defaults."""
        skill = SkillDefinition(name="test", description="A test.")
        assert skill.system_prompt_additions == ""
        assert skill.tool_subset is None
        assert skill.input_schema == {}
        assert skill.examples is None
        assert skill.max_steps == 15
