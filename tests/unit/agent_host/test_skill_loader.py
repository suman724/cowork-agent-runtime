"""Tests for SkillLoader — built-in markdown and user directory-based skill loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_host.skills.models import SkillDefinition
from agent_host.skills.skill_loader import (
    SkillLoader,
    _split_frontmatter,
)


class TestFrontmatterParser:
    def test_valid_frontmatter(self) -> None:
        """Should parse valid YAML frontmatter and return (dict, body)."""
        text = "---\nname: test\ndescription: A test.\n---\nBody content here."
        metadata, body = _split_frontmatter(text)
        assert metadata == {"name": "test", "description": "A test."}
        assert body.strip() == "Body content here."

    def test_no_frontmatter(self) -> None:
        """Text without frontmatter returns (None, original text)."""
        text = "No frontmatter here."
        metadata, body = _split_frontmatter(text)
        assert metadata is None
        assert body == text

    def test_invalid_yaml(self) -> None:
        """Invalid YAML in frontmatter returns (None, original text)."""
        text = "---\n: : :\n  bad yaml [[\n---\nBody."
        metadata, body = _split_frontmatter(text)
        assert metadata is None
        assert body == text

    def test_empty_string(self) -> None:
        """Empty string returns (None, '')."""
        metadata, body = _split_frontmatter("")
        assert metadata is None
        assert body == ""

    def test_frontmatter_without_closing(self) -> None:
        """Unclosed frontmatter returns (None, original text)."""
        text = "---\nname: test\nno closing delimiter"
        metadata, body = _split_frontmatter(text)
        assert metadata is None
        assert body == text

    def test_non_dict_frontmatter(self) -> None:
        """Frontmatter that parses to non-dict returns (None, original text)."""
        text = "---\n- item1\n- item2\n---\nBody."
        metadata, _body = _split_frontmatter(text)
        assert metadata is None

    def test_leading_whitespace(self) -> None:
        """Leading whitespace before frontmatter should be tolerated."""
        text = "\n  ---\nname: test\ndescription: A test.\n---\nBody."
        metadata, _body = _split_frontmatter(text)
        assert metadata is not None
        assert metadata["name"] == "test"


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
        assert search.prompt_content  # pre-populated for built-in

        edit = skills["edit_and_verify"]
        assert edit.tool_subset == ["ReadFile", "WriteFile", "RunCommand"]

        debug = skills["debug_error"]
        assert debug.tool_subset == ["ReadFile", "RunCommand"]

    def test_builtin_prompt_content_prepopulated(self) -> None:
        """Built-in skills should have prompt_content pre-populated (no lazy loading)."""
        loader = SkillLoader()
        skills = {s.name: s for s in loader.load_all()}

        builtin_names = {"search_codebase", "edit_and_verify", "debug_error"}
        for name in builtin_names:
            skill = skills[name]
            assert skill.prompt_content, f"Built-in skill '{name}' has empty prompt_content"
            assert skill.source_dir is None

    def test_builtin_defaults(self) -> None:
        """Built-in skills should have correct defaults for new fields."""
        loader = SkillLoader()
        skills = {s.name: s for s in loader.load_all()}

        builtin_names = {"search_codebase", "edit_and_verify", "debug_error"}
        for name in builtin_names:
            skill = skills[name]
            assert skill.disable_model_invocation is False
            assert skill.user_invocable is True


class TestUserSkills:
    def test_load_user_skill_from_directory(self, tmp_path: Path) -> None:
        """Should load a skill from a directory with SKILL.md."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "deploy-staging"
        skill_dir.mkdir(parents=True)

        skill_md = """\
---
name: deploy_staging
description: Deploy the application to staging environment.
tool_subset:
  - RunCommand
max_steps: 20
---

Deploy to staging using the configured pipeline.
"""
        (skill_dir / "SKILL.md").write_text(skill_md)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        assert "deploy_staging" in skills
        deploy = skills["deploy_staging"]
        assert deploy.description == "Deploy the application to staging environment."
        assert deploy.tool_subset == ["RunCommand"]
        assert deploy.max_steps == 20
        assert deploy.source_dir == str(skill_dir)
        # Stage 1: prompt_content is empty (not yet loaded)
        assert deploy.prompt_content == ""

    def test_name_derived_from_directory(self, tmp_path: Path) -> None:
        """Name should be derived from directory name when not in frontmatter."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-custom-tool"
        skill_dir.mkdir(parents=True)

        skill_md = """\
---
description: A custom tool skill.
---

Instructions for the custom tool.
"""
        (skill_dir / "SKILL.md").write_text(skill_md)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        assert "my_custom_tool" in skills

    def test_user_overrides_builtin(self, tmp_path: Path) -> None:
        """User skill with same name should override built-in."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "search-codebase"
        skill_dir.mkdir(parents=True)

        skill_md = """\
---
name: search_codebase
description: Custom search override.
max_steps: 25
---

Custom search instructions.
"""
        (skill_dir / "SKILL.md").write_text(skill_md)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        assert skills["search_codebase"].description == "Custom search override."
        assert skills["search_codebase"].max_steps == 25

    def test_missing_description_skipped(self, tmp_path: Path) -> None:
        """Skill with missing description in frontmatter should be skipped."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "bad-skill"
        skill_dir.mkdir(parents=True)

        skill_md = """\
---
name: bad_skill
---

No description means this is skipped.
"""
        (skill_dir / "SKILL.md").write_text(skill_md)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}
        assert "bad_skill" not in skills

    def test_invalid_frontmatter_skipped(self, tmp_path: Path) -> None:
        """Invalid frontmatter should cause skill to be skipped."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "broken"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text("No frontmatter at all.")

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}
        assert "broken" not in skills

    def test_no_user_skills_dir(self, tmp_path: Path) -> None:
        """Non-existent user skills dir should return only built-in skills."""
        loader = SkillLoader(user_skills_dir=str(tmp_path / "nonexistent"))
        skills = loader.load_all()
        assert len(skills) == 3  # only built-ins

    def test_directory_without_skill_md_skipped(self, tmp_path: Path) -> None:
        """Directory without SKILL.md should be skipped."""
        skills_dir = tmp_path / "skills"
        (skills_dir / "empty-dir").mkdir(parents=True)
        (skills_dir / "has-other-file").mkdir(parents=True)
        (skills_dir / "has-other-file" / "readme.md").write_text("Not a SKILL.md")

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = loader.load_all()
        assert len(skills) == 3  # only built-ins

    def test_new_fields_from_frontmatter(self, tmp_path: Path) -> None:
        """Should parse disable_model_invocation and user_invocable from frontmatter."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "hidden-skill"
        skill_dir.mkdir(parents=True)

        skill_md = """\
---
name: hidden_skill
description: A hidden skill.
disable_model_invocation: true
user_invocable: false
---

Secret instructions.
"""
        (skill_dir / "SKILL.md").write_text(skill_md)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        hidden = skills["hidden_skill"]
        assert hidden.disable_model_invocation is True
        assert hidden.user_invocable is False


class TestProgressiveDisclosure:
    def test_metadata_only_has_empty_prompt_content(self, tmp_path: Path) -> None:
        """Stage 1: User skill loaded with empty prompt_content."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir(parents=True)

        skill_md = """\
---
name: test_skill
description: A test skill.
---

Some instructions here.
"""
        (skill_dir / "SKILL.md").write_text(skill_md)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        assert skills["test_skill"].prompt_content == ""

    def test_load_skill_content_populates(self, tmp_path: Path) -> None:
        """Stage 2: load_skill_content() populates prompt_content from SKILL.md body."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir(parents=True)

        skill_md = """\
---
name: test_skill
description: A test skill.
---

Some instructions here.
"""
        (skill_dir / "SKILL.md").write_text(skill_md)

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        loaded = SkillLoader.load_skill_content(skills["test_skill"])
        assert loaded.prompt_content == "Some instructions here."

    def test_builtin_already_populated(self) -> None:
        """Built-in skills already have prompt_content — load_skill_content is a no-op."""
        loader = SkillLoader()
        skills = {s.name: s for s in loader.load_all()}

        search = skills["search_codebase"]
        original_content = search.prompt_content
        assert original_content  # pre-populated

        loaded = SkillLoader.load_skill_content(search)
        assert loaded.prompt_content == original_content


class TestSupportingFiles:
    def test_supporting_files_appended(self, tmp_path: Path) -> None:
        """Supporting .md files should be appended with section headers."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "deploy"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            "---\nname: deploy\ndescription: Deploy skill.\n---\n\nMain instructions."
        )
        (skill_dir / "examples.md").write_text("Example 1: do this.")
        (skill_dir / "reference.md").write_text("API reference details.")

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        loaded = SkillLoader.load_skill_content(skills["deploy"])

        assert "Main instructions." in loaded.prompt_content
        assert "## Examples" in loaded.prompt_content
        assert "Example 1: do this." in loaded.prompt_content
        assert "## Reference" in loaded.prompt_content
        assert "API reference details." in loaded.prompt_content

    def test_supporting_files_sorted_alphabetically(self, tmp_path: Path) -> None:
        """Supporting files should be appended in sorted order."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "ordered"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            "---\nname: ordered\ndescription: Test ordering.\n---\n\nMain."
        )
        (skill_dir / "c-file.md").write_text("Content C.")
        (skill_dir / "a-file.md").write_text("Content A.")
        (skill_dir / "b-file.md").write_text("Content B.")

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        loaded = SkillLoader.load_skill_content(skills["ordered"])

        # A should come before B, B before C
        pos_a = loaded.prompt_content.index("Content A.")
        pos_b = loaded.prompt_content.index("Content B.")
        pos_c = loaded.prompt_content.index("Content C.")
        assert pos_a < pos_b < pos_c

    def test_skill_md_body_comes_first(self, tmp_path: Path) -> None:
        """SKILL.md body should come before supporting files."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "body-first"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            "---\nname: body_first\ndescription: Test body first.\n---\n\nMain body content."
        )
        (skill_dir / "extra.md").write_text("Extra content.")

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        loaded = SkillLoader.load_skill_content(skills["body_first"])

        pos_main = loaded.prompt_content.index("Main body content.")
        pos_extra = loaded.prompt_content.index("Extra content.")
        assert pos_main < pos_extra

    def test_separator_between_body_and_supporting(self, tmp_path: Path) -> None:
        """A --- separator should appear between body and supporting files."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "sep"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            "---\nname: sep\ndescription: Test separator.\n---\n\nBody."
        )
        (skill_dir / "notes.md").write_text("Some notes.")

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        loaded = SkillLoader.load_skill_content(skills["sep"])
        assert "\n\n---\n\n" in loaded.prompt_content

    def test_non_md_files_ignored(self, tmp_path: Path) -> None:
        """Non-.md files in skill directory should be ignored."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "mixed"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            "---\nname: mixed\ndescription: Mixed files.\n---\n\nBody."
        )
        (skill_dir / "data.json").write_text('{"key": "value"}')
        (skill_dir / "script.py").write_text("print('hello')")

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        loaded = SkillLoader.load_skill_content(skills["mixed"])
        assert "key" not in loaded.prompt_content
        assert "hello" not in loaded.prompt_content

    def test_empty_supporting_files_skipped(self, tmp_path: Path) -> None:
        """Empty supporting files should not produce sections."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "empty-support"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            "---\nname: empty_support\ndescription: Test empty.\n---\n\nBody."
        )
        (skill_dir / "empty.md").write_text("")
        (skill_dir / "whitespace.md").write_text("   \n  \n  ")

        loader = SkillLoader(user_skills_dir=str(skills_dir))
        skills = {s.name: s for s in loader.load_all()}

        loaded = SkillLoader.load_skill_content(skills["empty_support"])
        assert "## Empty" not in loaded.prompt_content
        assert "## Whitespace" not in loaded.prompt_content
        assert "---\n\n" not in loaded.prompt_content  # no separator if no supporting content


class TestPolicySkills:
    def test_load_policy_skills(self) -> None:
        """Should load skills from policy bundle data."""
        policy_skills: list[dict[str, Any]] = [
            {
                "name": "policy_scan",
                "description": "Run a security scan.",
                "promptContent": "You are a security scanner.",
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
        assert scan.prompt_content == "You are a security scanner."
        assert scan.tool_subset == ["RunCommand"]
        assert scan.max_steps == 10

    def test_policy_overrides_user(self, tmp_path: Path) -> None:
        """Policy skill should override user skill with same name."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "shared-skill"
        skill_dir.mkdir(parents=True)

        skill_md = """\
---
name: shared_skill
description: From user.
max_steps: 10
---

User instructions.
"""
        (skill_dir / "SKILL.md").write_text(skill_md)

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
        assert minimal.prompt_content == ""
        assert minimal.tool_subset is None
        assert minimal.max_steps == 15

    def test_policy_skill_new_fields(self) -> None:
        """Policy skills should support disable_model_invocation and user_invocable."""
        policy_skills: list[dict[str, Any]] = [
            {
                "name": "restricted_skill",
                "description": "Restricted.",
                "disableModelInvocation": True,
                "userInvocable": False,
            }
        ]

        loader = SkillLoader(policy_skills=policy_skills)
        skills = {s.name: s for s in loader.load_all()}

        restricted = skills["restricted_skill"]
        assert restricted.disable_model_invocation is True
        assert restricted.user_invocable is False


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
        assert skill.prompt_content == ""
        assert skill.source_dir is None
        assert skill.tool_subset is None
        assert skill.input_schema == {}
        assert skill.max_steps == 15
        assert skill.disable_model_invocation is False
        assert skill.user_invocable is True
