"""Tests for SystemPromptBuilder — static prompt, workspace detection."""

from __future__ import annotations

from pathlib import Path

from agent_host.loop.system_prompt import SystemPromptBuilder


class TestSystemPromptBuilder:
    def test_static_prompt_includes_identity(self) -> None:
        builder = SystemPromptBuilder()
        prompt = builder.build_static_prompt()
        assert "Cowork" in prompt
        assert "tools" in prompt

    def test_static_prompt_includes_current_date(self) -> None:
        from datetime import UTC, datetime

        builder = SystemPromptBuilder()
        prompt = builder.build_static_prompt()
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        assert today in prompt

    def test_static_prompt_includes_os(self) -> None:
        builder = SystemPromptBuilder(os_family="Darwin")
        prompt = builder.build_static_prompt()
        assert "Darwin" in prompt

    def test_workspace_detection_with_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / ".git").mkdir()
        builder = SystemPromptBuilder(workspace_dir=str(tmp_path))
        prompt = builder.build_static_prompt()
        assert "Python project" in prompt
        assert "Git repository" in prompt

    def test_workspace_detection_with_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").touch()
        builder = SystemPromptBuilder(workspace_dir=str(tmp_path))
        prompt = builder.build_static_prompt()
        assert "Node.js" in prompt

    def test_workspace_detection_no_dir(self) -> None:
        builder = SystemPromptBuilder(workspace_dir=None)
        prompt = builder.build_static_prompt()
        assert "Workspace" not in prompt or "Directory" not in prompt

    def test_workspace_detection_nonexistent_dir(self) -> None:
        builder = SystemPromptBuilder(workspace_dir="/nonexistent/path/xyz")
        prompt = builder.build_static_prompt()
        # Should not crash, just no workspace context
        assert "Cowork" in prompt

    def test_dynamic_injection_empty(self) -> None:
        builder = SystemPromptBuilder()
        injection = builder.build_dynamic_injection()
        assert injection == ""

    def test_static_prompt_includes_project_instructions(self) -> None:
        builder = SystemPromptBuilder()
        prompt = builder.build_static_prompt(project_instructions="Use pytest for all tests.")
        assert "# Project Instructions" in prompt
        assert "Use pytest for all tests." in prompt

    def test_static_prompt_without_project_instructions(self) -> None:
        builder = SystemPromptBuilder()
        prompt = builder.build_static_prompt(project_instructions="")
        assert "# Project Instructions" not in prompt

    def test_memory_guidance_when_persistent_memory(self) -> None:
        builder = SystemPromptBuilder()
        prompt = builder.build_static_prompt(has_persistent_memory=True)
        assert "# Memory" in prompt
        assert "SaveMemory" in prompt
        assert "RecallMemory" in prompt

    def test_no_memory_guidance_without_persistent_memory(self) -> None:
        builder = SystemPromptBuilder()
        prompt = builder.build_static_prompt(has_persistent_memory=False)
        assert "# Memory" not in prompt
