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


class TestCheckpointManagerAtomicWrite:
    def test_creates_dir_if_needed(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "nested" / "dir"
        CheckpointManager(str(new_dir))
        assert new_dir.exists()
