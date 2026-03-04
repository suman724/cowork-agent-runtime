"""Tests for PersistentMemory — AI-writable memory files."""

from __future__ import annotations

from pathlib import Path

from agent_host.memory.persistent_memory import PersistentMemory


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
        assert "Saved" in result
        assert "notes.md" in result

        content = pm.read_file("notes.md")
        assert content == "Hello world"

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        pm.save_file("notes.md", "Version 1")
        pm.save_file("notes.md", "Version 2")
        assert pm.read_file("notes.md") == "Version 2"

    def test_read_missing_file(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        result = pm.read_file("nonexistent.md")
        assert result.startswith("Error")

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "memory"
        pm = PersistentMemory(str(nested))
        result = pm.save_file("test.md", "content")
        assert "Saved" in result
        assert nested.is_dir()


class TestFilenameValidation:
    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert pm.save_file("../escape.md", "bad").startswith("Error")
        assert pm.read_file("../escape.md").startswith("Error")

    def test_rejects_slashes(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert pm.save_file("sub/file.md", "bad").startswith("Error")
        assert pm.save_file("sub\\file.md", "bad").startswith("Error")

    def test_rejects_non_md(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert pm.save_file("notes.txt", "bad").startswith("Error")
        assert pm.save_file("script.py", "bad").startswith("Error")

    def test_rejects_empty_filename(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert pm.save_file("", "content").startswith("Error")

    def test_allows_valid_names(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        pm.ensure_dir()
        assert "Saved" in pm.save_file("MEMORY.md", "ok")
        assert "Saved" in pm.save_file("debugging.md", "ok")
        assert "Saved" in pm.save_file("api-patterns.md", "ok")
        assert "Saved" in pm.save_file("notes_v2.md", "ok")


class TestListFiles:
    def test_empty_directory(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert pm.list_files() == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path / "nope"))
        assert pm.list_files() == []

    def test_lists_md_files_only(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("index")
        (tmp_path / "notes.md").write_text("notes")
        (tmp_path / "temp.tmp").write_text("temp")
        pm = PersistentMemory(str(tmp_path))
        files = pm.list_files()
        names = [f["name"] for f in files]
        assert "MEMORY.md" in names
        assert "notes.md" in names
        assert "temp.tmp" not in names

    def test_includes_file_sizes(self, tmp_path: Path) -> None:
        (tmp_path / "test.md").write_text("12345")
        pm = PersistentMemory(str(tmp_path))
        files = pm.list_files()
        assert len(files) == 1
        assert files[0]["name"] == "test.md"
        assert files[0]["size"] == 5
