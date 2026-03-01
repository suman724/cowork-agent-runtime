"""Tests for FileChangeTracker — file mutation tracking for patch preview."""

from __future__ import annotations

from agent_host.agent.file_change_tracker import FileChangeTracker


class TestFileChangeTracker:
    def test_record_write_new_file(self) -> None:
        """Writing a new file (old_content=None) records status='added'."""
        tracker = FileChangeTracker()
        tracker.record_write("task-1", "/tmp/new.txt", old_content=None, new_content="hello")

        preview = tracker.get_patch_preview("task-1")
        assert preview["taskId"] == "task-1"
        files = preview["files"]
        assert len(files) == 1
        assert files[0]["status"] == "added"
        assert files[0]["filePath"] == "/tmp/new.txt"

    def test_record_write_existing_file(self) -> None:
        """Overwriting an existing file records status='modified'."""
        tracker = FileChangeTracker()
        tracker.record_write("task-1", "/tmp/existing.txt", old_content="old", new_content="new")

        preview = tracker.get_patch_preview("task-1")
        files = preview["files"]
        assert len(files) == 1
        assert files[0]["status"] == "modified"
        assert "old" in files[0]["hunks"]
        assert "new" in files[0]["hunks"]

    def test_record_delete(self) -> None:
        """Deleting a file records status='deleted'."""
        tracker = FileChangeTracker()
        tracker.record_delete("task-1", "/tmp/gone.txt", old_content="goodbye")

        preview = tracker.get_patch_preview("task-1")
        files = preview["files"]
        assert len(files) == 1
        assert files[0]["status"] == "deleted"
        assert files[0]["filePath"] == "/tmp/gone.txt"

    def test_get_patch_preview_format(self) -> None:
        """Verify returned dict matches expected PatchPreview shape."""
        tracker = FileChangeTracker()
        tracker.record_write("task-1", "/a.txt", old_content=None, new_content="line1\nline2\n")
        tracker.record_write("task-1", "/b.txt", old_content="old\n", new_content="new\n")
        tracker.record_delete("task-1", "/c.txt", old_content="content\n")

        preview = tracker.get_patch_preview("task-1")
        assert "taskId" in preview
        assert "files" in preview
        files = preview["files"]
        assert len(files) == 3

        # Verify each file has required keys
        for f in files:
            assert "filePath" in f
            assert "status" in f
            assert "hunks" in f
            assert f["status"] in ("added", "modified", "deleted")

    def test_clear_task(self) -> None:
        """Changes are removed after clear_task."""
        tracker = FileChangeTracker()
        tracker.record_write("task-1", "/tmp/a.txt", old_content=None, new_content="data")

        tracker.clear_task("task-1")

        preview = tracker.get_patch_preview("task-1")
        assert preview["files"] == []

    def test_empty_task_returns_empty_files(self) -> None:
        """get_patch_preview for unknown task returns empty files list."""
        tracker = FileChangeTracker()
        preview = tracker.get_patch_preview("nonexistent")
        assert preview["taskId"] == "nonexistent"
        assert preview["files"] == []
