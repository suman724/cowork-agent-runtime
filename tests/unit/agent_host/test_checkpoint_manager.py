"""Tests for CheckpointManager — save/load/delete, atomic write, corruption."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_host.session.checkpoint_manager import CheckpointManager, SessionCheckpoint


@pytest.fixture
def manager(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(str(tmp_path))


def _make_checkpoint(**kwargs: object) -> SessionCheckpoint:
    defaults = {
        "session_id": "sess-123",
        "workspace_id": "ws-456",
        "tenant_id": "tenant-test",
        "user_id": "user-test",
        "token_input_used": 100,
        "token_output_used": 50,
        "session_messages": [{"messageId": "m1", "role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return SessionCheckpoint(**defaults)  # type: ignore[arg-type]


class TestCheckpointManagerSaveLoad:
    def test_save_and_load(self, manager: CheckpointManager) -> None:
        checkpoint = _make_checkpoint()
        manager.save(checkpoint)

        loaded = manager.load("sess-123")
        assert loaded is not None
        assert loaded.session_id == "sess-123"
        assert loaded.workspace_id == "ws-456"
        assert loaded.token_input_used == 100
        assert loaded.token_output_used == 50
        assert len(loaded.session_messages) == 1
        assert loaded.checkpointed_at != ""

    def test_load_nonexistent(self, manager: CheckpointManager) -> None:
        assert manager.load("nonexistent") is None

    def test_overwrite_checkpoint(self, manager: CheckpointManager) -> None:
        checkpoint = _make_checkpoint(token_input_used=100)
        manager.save(checkpoint)

        checkpoint2 = _make_checkpoint(token_input_used=200)
        manager.save(checkpoint2)

        loaded = manager.load("sess-123")
        assert loaded is not None
        assert loaded.token_input_used == 200


class TestCheckpointManagerDelete:
    def test_delete(self, manager: CheckpointManager) -> None:
        checkpoint = _make_checkpoint()
        manager.save(checkpoint)
        manager.delete("sess-123")
        assert manager.load("sess-123") is None

    def test_delete_nonexistent(self, manager: CheckpointManager) -> None:
        # Should not raise
        manager.delete("nonexistent")


class TestCheckpointManagerCorruption:
    def test_corrupt_json(self, manager: CheckpointManager) -> None:
        # Write corrupt data
        path = manager._checkpoint_path("sess-bad")
        path.write_text("not valid json {{{")

        loaded = manager.load("sess-bad")
        assert loaded is None
        # File should be deleted
        assert not path.exists()

    def test_missing_required_field(self, manager: CheckpointManager) -> None:
        path = manager._checkpoint_path("sess-bad")
        # Write data missing session_id
        path.write_text(json.dumps({"workspace_id": "ws"}))

        loaded = manager.load("sess-bad")
        assert loaded is None


class TestCheckpointManagerThread:
    def test_save_and_load_with_thread(self, manager: CheckpointManager) -> None:
        thread_data = [
            {"system_prompt": "test"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        checkpoint = _make_checkpoint(thread=thread_data)
        manager.save(checkpoint)

        loaded = manager.load("sess-123")
        assert loaded is not None
        assert loaded.thread == thread_data

    def test_save_and_load_without_thread(self, manager: CheckpointManager) -> None:
        checkpoint = _make_checkpoint(thread=None)
        manager.save(checkpoint)

        loaded = manager.load("sess-123")
        assert loaded is not None
        assert loaded.thread is None


class TestCheckpointManagerThreadRoundTrip:
    def test_thread_restored_matches_original(self, manager: CheckpointManager) -> None:
        """Thread data should survive checkpoint round-trip exactly."""
        from agent_host.llm.models import ToolCallMessage
        from agent_host.thread.message_thread import MessageThread

        # Build a realistic thread
        thread = MessageThread(system_prompt="You are helpful.")
        thread.add_user_message("Hello")
        thread.add_assistant_message(
            "",
            [ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/foo"})],
        )
        thread.add_tool_result("tc1", "ReadFile", '{"status": "success", "output": "contents"}')
        thread.add_assistant_message("Here is the file content.")
        thread.add_user_message("Thanks, now write a new file.")
        thread.add_assistant_message("Sure!")

        # Save to checkpoint
        checkpoint = _make_checkpoint(thread=thread.to_checkpoint())
        manager.save(checkpoint)

        # Load and restore
        loaded = manager.load("sess-123")
        assert loaded is not None
        assert loaded.thread is not None

        restored = MessageThread.from_checkpoint(loaded.thread)
        assert restored.get_system_prompt() == "You are helpful."
        assert restored.message_count == 6
        assert restored.messages[0]["role"] == "user"
        assert restored.messages[1]["role"] == "assistant"
        assert restored.messages[2]["role"] == "tool"
        assert restored.messages[3]["role"] == "assistant"
        assert restored.messages[4]["role"] == "user"
        assert restored.messages[5]["role"] == "assistant"

    def test_working_memory_field_persisted(self, manager: CheckpointManager) -> None:
        """Working memory dict should survive checkpoint round-trip."""
        wm_data = {
            "tasks": [{"id": "t1", "status": "in_progress", "content": "Implement auth"}],
            "notes": ["User prefers JWT"],
        }
        checkpoint = _make_checkpoint(working_memory=wm_data)
        manager.save(checkpoint)

        loaded = manager.load("sess-123")
        assert loaded is not None
        assert loaded.working_memory == wm_data


class TestCheckpointManagerActiveTask:
    def test_save_and_load_with_active_task(self, manager: CheckpointManager) -> None:
        """Active task fields should round-trip correctly."""
        checkpoint = _make_checkpoint(
            active_task_id="task-42",
            active_task_prompt="Fix the bug",
            active_task_step=7,
            active_task_max_steps=50,
            last_workspace_sync_step=5,
        )
        manager.save(checkpoint)

        loaded = manager.load("sess-123")
        assert loaded is not None
        assert loaded.active_task_id == "task-42"
        assert loaded.active_task_prompt == "Fix the bug"
        assert loaded.active_task_step == 7
        assert loaded.active_task_max_steps == 50
        assert loaded.last_workspace_sync_step == 5

    def test_backward_compat_old_checkpoint_without_active_task(
        self, manager: CheckpointManager
    ) -> None:
        """Loading a checkpoint written before active task fields were added."""
        path = manager._checkpoint_path("sess-old")
        old_data = {
            "session_id": "sess-old",
            "workspace_id": "ws-1",
            "tenant_id": "t-1",
            "user_id": "u-1",
            "token_input_used": 50,
            "token_output_used": 25,
            "session_messages": [],
            "checkpointed_at": "2026-01-01T00:00:00+00:00",
        }
        path.write_text(json.dumps(old_data))

        loaded = manager.load("sess-old")
        assert loaded is not None
        assert loaded.active_task_id is None
        assert loaded.active_task_prompt is None
        assert loaded.active_task_step == 0
        assert loaded.active_task_max_steps == 0
        assert loaded.last_workspace_sync_step == 0

    def test_active_task_cleared_on_task_end(self, manager: CheckpointManager) -> None:
        """When task ends, checkpoint should have None/0 for active task fields."""
        # First save with active task
        checkpoint = _make_checkpoint(
            active_task_id="task-1",
            active_task_step=10,
        )
        manager.save(checkpoint)

        # Then save without active task (task completed)
        checkpoint2 = _make_checkpoint(
            active_task_id=None,
            active_task_step=0,
        )
        manager.save(checkpoint2)

        loaded = manager.load("sess-123")
        assert loaded is not None
        assert loaded.active_task_id is None
        assert loaded.active_task_step == 0


class TestCheckpointManagerWorkspaceDir:
    def test_workspace_dir_round_trip(self, manager: CheckpointManager) -> None:
        """workspace_dir should survive checkpoint save/load."""
        checkpoint = _make_checkpoint(workspace_dir="/home/user/project")
        manager.save(checkpoint)

        loaded = manager.load("sess-123")
        assert loaded is not None
        assert loaded.workspace_dir == "/home/user/project"

    def test_old_checkpoint_without_workspace_dir(self, manager: CheckpointManager) -> None:
        """Loading an old checkpoint without workspace_dir should default to None."""
        path = manager._checkpoint_path("sess-old")
        old_data = {
            "session_id": "sess-old",
            "workspace_id": "ws-1",
            "tenant_id": "t-1",
            "user_id": "u-1",
            "token_input_used": 50,
            "token_output_used": 25,
            "session_messages": [],
            "checkpointed_at": "2026-01-01T00:00:00+00:00",
        }
        path.write_text(json.dumps(old_data))

        loaded = manager.load("sess-old")
        assert loaded is not None
        assert loaded.workspace_dir is None


class TestCheckpointManagerAtomicWrite:
    def test_creates_dir_if_needed(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "nested" / "dir"
        CheckpointManager(str(new_dir))
        assert new_dir.exists()
