"""Tests for ProjectInstructionsLoader — load COWORK.md from workspace directory."""

from __future__ import annotations

from pathlib import Path

from agent_host.memory.project_instructions import ProjectInstructionsLoader


class TestProjectInstructionsLoader:
    def test_no_files_returns_empty(self, tmp_path: Path) -> None:
        loader = ProjectInstructionsLoader()
        assert loader.load(str(tmp_path)) == ""

    def test_single_cowork_md(self, tmp_path: Path) -> None:
        (tmp_path / "COWORK.md").write_text("# My Project\nUse pytest.")
        loader = ProjectInstructionsLoader()
        result = loader.load(str(tmp_path))
        assert "# My Project" in result
        assert "Use pytest." in result

    def test_local_md_appended_after_cowork_md(self, tmp_path: Path) -> None:
        (tmp_path / "COWORK.md").write_text("Team instructions")
        (tmp_path / "COWORK.local.md").write_text("My local notes")
        loader = ProjectInstructionsLoader()
        result = loader.load(str(tmp_path))

        # Both present
        assert "Team instructions" in result
        assert "My local notes" in result

        # COWORK.md appears before COWORK.local.md
        team_pos = result.index("Team instructions")
        local_pos = result.index("My local notes")
        assert team_pos < local_pos

    def test_does_not_walk_up_directory_tree(self, tmp_path: Path) -> None:
        """Files in ancestor directories are NOT collected (no ancestor walk)."""
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)

        (parent / "COWORK.md").write_text("Parent rules")
        (child / "COWORK.md").write_text("Child rules")

        loader = ProjectInstructionsLoader()
        result = loader.load(str(child))

        # Only child directory files should be loaded
        assert "Child rules" in result
        assert "Parent rules" not in result

    def test_empty_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "COWORK.md").write_text("")
        (tmp_path / "COWORK.local.md").write_text("Local notes")
        loader = ProjectInstructionsLoader()
        result = loader.load(str(tmp_path))
        # Empty COWORK.md skipped, local.md present
        assert "Local notes" in result

    def test_nonexistent_dir_returns_empty(self) -> None:
        loader = ProjectInstructionsLoader()
        assert loader.load("/nonexistent/path/xyz") == ""

    def test_only_local_md(self, tmp_path: Path) -> None:
        (tmp_path / "COWORK.local.md").write_text("Only local")
        loader = ProjectInstructionsLoader()
        result = loader.load(str(tmp_path))
        assert "Only local" in result

    def test_no_separator_in_output(self, tmp_path: Path) -> None:
        """Output should not contain --- separators (simplified loader)."""
        (tmp_path / "COWORK.md").write_text("content")
        loader = ProjectInstructionsLoader()
        result = loader.load(str(tmp_path))
        assert "---" not in result
