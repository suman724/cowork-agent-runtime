"""Tests for PersistentMemory — AI-writable memory files."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from agent_host.memory.persistent_memory import (
    MAX_FILENAME_LENGTH,
    MemoryResult,
    PersistentMemory,
    _build_memory_md,
    _parse_auto_index,
)


class TestResolveMemoryDir:
    def test_deterministic(self) -> None:
        """Same workspace_dir always produces the same memory dir."""
        dir1 = PersistentMemory.resolve_memory_dir("/home/user/project")
        dir2 = PersistentMemory.resolve_memory_dir("/home/user/project")
        assert dir1 == dir2

    def test_different_paths_different_dirs(self) -> None:
        dir1 = PersistentMemory.resolve_memory_dir("/home/user/project-a")
        dir2 = PersistentMemory.resolve_memory_dir("/home/user/project-b")
        assert dir1 != dir2

    def test_ends_with_memory(self) -> None:
        result = PersistentMemory.resolve_memory_dir("/some/path")
        assert result.endswith("/memory") or result.endswith("\\memory")


class TestLoadIndex:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert pm.load_index() == ""

    def test_loads_content(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("# Memory\nLine 1\nLine 2")
        pm = PersistentMemory(str(tmp_path))
        result = pm.load_index()
        assert "# Memory" in result
        assert "Line 2" in result

    def test_respects_max_lines(self, tmp_path: Path) -> None:
        lines = [f"Line {i}" for i in range(300)]
        (tmp_path / "MEMORY.md").write_text("\n".join(lines))
        pm = PersistentMemory(str(tmp_path))
        result = pm.load_index(max_lines=10)
        loaded_lines = result.splitlines()
        assert len(loaded_lines) == 10
        assert loaded_lines[0] == "Line 0"
        assert loaded_lines[9] == "Line 9"

    def test_default_max_200_lines(self, tmp_path: Path) -> None:
        lines = [f"Line {i}" for i in range(250)]
        (tmp_path / "MEMORY.md").write_text("\n".join(lines))
        pm = PersistentMemory(str(tmp_path))
        result = pm.load_index()
        loaded_lines = result.splitlines()
        assert len(loaded_lines) == 200


class TestSaveAndReadFile:
    def test_save_and_read(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        result = pm.save_file("notes.md", "Hello world")
        assert result.success
        assert "Saved" in result.content
        assert "notes.md" in result.content

        read_result = pm.read_file("notes.md")
        assert read_result.success
        assert read_result.content == "Hello world"

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        pm.save_file("notes.md", "Version 1")
        pm.save_file("notes.md", "Version 2")
        result = pm.read_file("notes.md")
        assert result.success
        assert result.content == "Version 2"

    def test_read_missing_file(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        result = pm.read_file("nonexistent.md")
        assert not result.success
        assert "not found" in result.content

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "memory"
        pm = PersistentMemory(str(nested))
        result = pm.save_file("test.md", "content")
        assert result.success
        assert "Saved" in result.content
        assert nested.is_dir()

    def test_returns_memory_result(self, tmp_path: Path) -> None:
        """All methods return MemoryResult (structured, not string prefix)."""
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        save_result = pm.save_file("test.md", "content")
        assert isinstance(save_result, MemoryResult)

        read_result = pm.read_file("test.md")
        assert isinstance(read_result, MemoryResult)


class TestFilenameValidation:
    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert not pm.save_file("../escape.md", "bad").success
        assert not pm.read_file("../escape.md").success

    def test_rejects_slashes(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert not pm.save_file("sub/file.md", "bad").success
        assert not pm.save_file("sub\\file.md", "bad").success

    def test_rejects_non_md(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert not pm.save_file("notes.txt", "bad").success
        assert not pm.save_file("script.py", "bad").success

    def test_rejects_empty_filename(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert not pm.save_file("", "content").success

    def test_allows_valid_names(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        assert pm.save_file("MEMORY.md", "ok").success
        assert pm.save_file("debugging.md", "ok").success
        assert pm.save_file("api-patterns.md", "ok").success
        assert pm.save_file("notes_v2.md", "ok").success

    def test_rejects_filename_exceeding_max_length(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        long_name = "a" * (MAX_FILENAME_LENGTH + 1) + ".md"
        result = pm.save_file(long_name, "content")
        assert not result.success
        assert "too long" in result.content


class TestResourceLimits:
    def test_rejects_content_exceeding_max_file_size(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path), max_file_size=100)
        pm.ensure_dir()
        result = pm.save_file("big.md", "x" * 200)
        assert not result.success
        assert "max file size" in result.content

    def test_rejects_when_max_file_count_reached(self, tmp_path: Path) -> None:
        # max_file_count=3: a.md + MEMORY.md (auto-created) + b.md = 3 files, c.md should fail
        pm = PersistentMemory(str(tmp_path), max_file_count=3)
        pm.ensure_dir()
        pm.save_file("a.md", "content")
        pm.save_file("b.md", "content")
        result = pm.save_file("c.md", "content")
        assert not result.success
        assert "Max file count" in result.content

    def test_allows_overwrite_at_max_count(self, tmp_path: Path) -> None:
        # max_file_count=3: a.md + MEMORY.md (auto-created) + b.md = 3 files
        pm = PersistentMemory(str(tmp_path), max_file_count=3)
        pm.ensure_dir()
        pm.save_file("a.md", "v1")
        pm.save_file("b.md", "v1")
        # Overwriting existing file should succeed even at limit
        result = pm.save_file("a.md", "v2")
        assert result.success


class TestDeleteFile:
    def test_delete_topic_file(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        pm.save_file("old-notes.md", "content")
        # Verify auto-index entry was created
        memory_md = (tmp_path / "MEMORY.md").read_text()
        assert "old-notes.md" in memory_md

        result = pm.delete_file("old-notes.md")
        assert result.success
        assert "Deleted" in result.content
        assert not (tmp_path / "old-notes.md").exists()
        # Verify auto-index entry was removed
        memory_md = (tmp_path / "MEMORY.md").read_text()
        assert "old-notes.md" not in memory_md

    def test_rejects_memory_md_deletion(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        pm.save_file("MEMORY.md", "index")
        result = pm.delete_file("MEMORY.md")
        assert not result.success
        assert "Cannot delete MEMORY.md" in result.content
        assert (tmp_path / "MEMORY.md").exists()

    def test_error_for_nonexistent_file(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        result = pm.delete_file("nonexistent.md")
        assert not result.success
        assert "not found" in result.content

    def test_returns_memory_result(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        result = pm.delete_file("nonexistent.md")
        assert isinstance(result, MemoryResult)

    def test_rejects_invalid_filename(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        result = pm.delete_file("../escape.md")
        assert not result.success


class TestListFiles:
    def test_empty_directory(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert pm.list_files() == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path / "nope"))
        assert pm.list_files() == []

    def test_lists_md_files_only(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        # Use save_file so auto-index stays consistent
        pm.save_file("MEMORY.md", "index")
        pm.save_file("notes.md", "notes")
        (tmp_path / "temp.tmp").write_text("temp")
        files = pm.list_files()
        names = [f["name"] for f in files]
        assert "MEMORY.md" in names
        assert "notes.md" in names
        assert "temp.tmp" not in names

    def test_includes_file_sizes(self, tmp_path: Path) -> None:
        (tmp_path / "test.md").write_text("12345")
        pm = PersistentMemory(str(tmp_path))
        files = pm.list_files()
        # test.md is manually created (no auto-index), so only 1 file
        assert any(f["name"] == "test.md" and f["size"] == 5 for f in files)


class TestAutoIndex:
    """Tests for auto-maintained topic file index in MEMORY.md."""

    def test_saving_topic_file_creates_memory_md_with_index(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.save_file("debugging.md", "some debug notes")
        memory_md = (tmp_path / "MEMORY.md").read_text()
        assert "<!-- AUTO-INDEX:START -->" in memory_md
        assert "<!-- AUTO-INDEX:END -->" in memory_md
        assert "debugging.md" in memory_md
        assert "bytes" in memory_md

    def test_saving_memory_md_does_not_create_auto_index_entry(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.save_file("MEMORY.md", "# My Notes\nManual content")
        content = (tmp_path / "MEMORY.md").read_text()
        assert content == "# My Notes\nManual content"
        assert "AUTO-INDEX" not in content

    def test_manual_content_preserved(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.save_file("MEMORY.md", "# My Notes\nImportant fact")
        pm.save_file("patterns.md", "Factory pattern")
        content = (tmp_path / "MEMORY.md").read_text()
        assert "# My Notes" in content
        assert "Important fact" in content
        assert "patterns.md" in content

    def test_deleting_topic_file_removes_index_entry(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.save_file("old.md", "obsolete")
        assert "old.md" in (tmp_path / "MEMORY.md").read_text()
        pm.delete_file("old.md")
        content = (tmp_path / "MEMORY.md").read_text()
        assert "old.md" not in content

    def test_deleting_last_topic_file_removes_auto_index_section(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.save_file("only.md", "content")
        pm.delete_file("only.md")
        # MEMORY.md should either not exist or have no auto-index section
        if (tmp_path / "MEMORY.md").exists():
            content = (tmp_path / "MEMORY.md").read_text()
            assert "AUTO-INDEX" not in content

    def test_deleting_last_file_with_manual_content_preserves_manual(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.save_file("MEMORY.md", "# Important\nKeep this")
        pm.save_file("temp.md", "temporary")
        pm.delete_file("temp.md")
        content = (tmp_path / "MEMORY.md").read_text()
        assert "# Important" in content
        assert "Keep this" in content
        assert "AUTO-INDEX" not in content

    def test_overwriting_topic_file_updates_size(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.save_file("notes.md", "short")
        content1 = (tmp_path / "MEMORY.md").read_text()
        pm.save_file("notes.md", "much longer content here for testing size update")
        content2 = (tmp_path / "MEMORY.md").read_text()
        # The size in bytes should differ
        assert content1 != content2
        assert "notes.md" in content2

    def test_multiple_files_sorted_alphabetically(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.save_file("zebra.md", "z")
        pm.save_file("alpha.md", "a")
        pm.save_file("middle.md", "m")
        content = (tmp_path / "MEMORY.md").read_text()
        alpha_pos = content.index("alpha.md")
        middle_pos = content.index("middle.md")
        zebra_pos = content.index("zebra.md")
        assert alpha_pos < middle_pos < zebra_pos

    def test_auto_index_failure_does_not_fail_save(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        with patch.object(pm, "_do_update_auto_index", side_effect=OSError("disk full")):
            result = pm.save_file("notes.md", "content")
        assert result.success
        assert (tmp_path / "notes.md").exists()

    def test_auto_index_failure_does_not_fail_delete(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.save_file("notes.md", "content")
        with patch.object(pm, "_do_update_auto_index", side_effect=OSError("disk full")):
            result = pm.delete_file("notes.md")
        assert result.success
        assert not (tmp_path / "notes.md").exists()

    def test_appends_to_existing_memory_md_without_auto_index(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        # Write MEMORY.md with manual content (no auto-index)
        (tmp_path / "MEMORY.md").write_text("# Manual Notes\nSome content\n")
        pm.save_file("topic.md", "topic content")
        content = (tmp_path / "MEMORY.md").read_text()
        assert "# Manual Notes" in content
        assert "topic.md" in content
        assert "AUTO-INDEX:START" in content

    def test_handles_malformed_auto_index(self, tmp_path: Path) -> None:
        """Start marker without end marker — should still parse and rebuild correctly."""
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        malformed = (
            "# Notes\n\n<!-- AUTO-INDEX:START -->\n## Topic Files\n- [old.md](old.md) - 5 bytes\n"
        )
        (tmp_path / "MEMORY.md").write_text(malformed)
        pm.save_file("new.md", "new content")
        content = (tmp_path / "MEMORY.md").read_text()
        assert "# Notes" in content
        assert "new.md" in content
        assert "old.md" in content
        assert "<!-- AUTO-INDEX:END -->" in content


class TestParseAutoIndex:
    """Tests for the _parse_auto_index helper."""

    def test_no_markers(self) -> None:
        manual, entries = _parse_auto_index("# Notes\nSome content\n")
        assert manual == "# Notes\nSome content\n"
        assert entries == {}

    def test_empty_content(self) -> None:
        manual, entries = _parse_auto_index("")
        assert manual == ""
        assert entries == {}

    def test_parses_entries(self) -> None:
        content = (
            "# Notes\n\n"
            "<!-- AUTO-INDEX:START -->\n"
            "## Topic Files\n"
            "- [a.md](a.md) - 100 bytes\n"
            "- [b.md](b.md) - 2,340 bytes\n"
            "<!-- AUTO-INDEX:END -->\n"
        )
        manual, entries = _parse_auto_index(content)
        assert "# Notes" in manual
        assert entries == {"a.md": 100, "b.md": 2340}


class TestBuildMemoryMd:
    """Tests for the _build_memory_md helper."""

    def test_manual_only(self) -> None:
        result = _build_memory_md("# Notes", {})
        assert result == "# Notes\n"
        assert "AUTO-INDEX" not in result

    def test_entries_only(self) -> None:
        result = _build_memory_md("", {"a.md": 100})
        assert "a.md" in result
        assert "AUTO-INDEX:START" in result

    def test_both(self) -> None:
        result = _build_memory_md("# Notes", {"a.md": 100})
        assert result.startswith("# Notes")
        assert "a.md" in result

    def test_empty(self) -> None:
        result = _build_memory_md("", {})
        assert result == ""
