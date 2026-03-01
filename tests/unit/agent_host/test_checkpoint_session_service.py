"""Tests for CheckpointSessionService — JSON file checkpoint persistence."""

from __future__ import annotations

import pytest
from google.adk.events import Event, EventActions

from agent_host.session.checkpoint_session_service import CheckpointSessionService


class TestCheckpointSessionService:
    @pytest.mark.asyncio
    async def test_create_session(self, tmp_path: object) -> None:
        """create_session creates a session and writes checkpoint."""
        service = CheckpointSessionService(str(tmp_path))
        session = await service.create_session(
            app_name="test", user_id="user-1", session_id="sess-1"
        )

        assert session.id == "sess-1"
        assert session.user_id == "user-1"
        assert session.app_name == "test"

        # Checkpoint file exists
        checkpoint = tmp_path / "test_sess-1.json"
        assert checkpoint.exists()

    @pytest.mark.asyncio
    async def test_get_session_from_memory(self, tmp_path: object) -> None:
        """get_session returns in-memory session."""
        service = CheckpointSessionService(str(tmp_path))
        await service.create_session(app_name="test", user_id="user-1", session_id="sess-1")

        session = await service.get_session(app_name="test", user_id="user-1", session_id="sess-1")
        assert session is not None
        assert session.id == "sess-1"

    @pytest.mark.asyncio
    async def test_get_session_from_checkpoint(self, tmp_path: object) -> None:
        """get_session falls back to checkpoint file when not in memory."""
        service = CheckpointSessionService(str(tmp_path))
        await service.create_session(app_name="test", user_id="user-1", session_id="sess-1")

        # Clear in-memory cache
        service._sessions.clear()

        session = await service.get_session(app_name="test", user_id="user-1", session_id="sess-1")
        assert session is not None
        assert session.id == "sess-1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, tmp_path: object) -> None:
        """get_session returns None for unknown session."""
        service = CheckpointSessionService(str(tmp_path))
        session = await service.get_session(
            app_name="test", user_id="user-1", session_id="nonexistent"
        )
        assert session is None

    @pytest.mark.asyncio
    async def test_delete_session(self, tmp_path: object) -> None:
        """delete_session removes from memory and disk."""
        service = CheckpointSessionService(str(tmp_path))
        await service.create_session(app_name="test", user_id="user-1", session_id="sess-1")

        await service.delete_session(app_name="test", user_id="user-1", session_id="sess-1")

        # Not in memory
        assert "sess-1" not in service._sessions
        # Checkpoint deleted
        assert not (tmp_path / "test_sess-1.json").exists()

    @pytest.mark.asyncio
    async def test_append_event_updates_state(self, tmp_path: object) -> None:
        """append_event applies state_delta and writes checkpoint."""
        service = CheckpointSessionService(str(tmp_path))
        session = await service.create_session(
            app_name="test", user_id="user-1", session_id="sess-1"
        )

        event = Event(
            invocation_id="inv-1",
            author="agent",
            actions=EventActions(state_delta={"key": "value"}),
        )

        await service.append_event(session, event)
        assert session.state["key"] == "value"
        assert len(session.events) == 1

    @pytest.mark.asyncio
    async def test_corrupt_checkpoint_deleted(self, tmp_path: object) -> None:
        """Corrupt checkpoint file is deleted and None returned."""
        service = CheckpointSessionService(str(tmp_path))

        # Write corrupt checkpoint
        checkpoint = tmp_path / "test_sess-corrupt.json"
        checkpoint.write_text("{invalid json")

        session = await service.get_session(
            app_name="test", user_id="user-1", session_id="sess-corrupt"
        )
        assert session is None
        assert not checkpoint.exists()

    @pytest.mark.asyncio
    async def test_binary_checkpoint_deleted(self, tmp_path: object) -> None:
        """Binary (non-UTF-8) checkpoint file is deleted and None returned."""
        service = CheckpointSessionService(str(tmp_path))

        # Write binary content that is not valid UTF-8
        checkpoint = tmp_path / "test_sess-binary.json"
        checkpoint.write_bytes(b"\x80\x81\x82\xff\xfe")

        session = await service.get_session(
            app_name="test", user_id="user-1", session_id="sess-binary"
        )
        assert session is None
        assert not checkpoint.exists()

    @pytest.mark.asyncio
    async def test_list_sessions(self, tmp_path: object) -> None:
        """list_sessions returns all in-memory sessions."""
        service = CheckpointSessionService(str(tmp_path))
        await service.create_session(app_name="test", user_id="user-1", session_id="sess-1")
        await service.create_session(app_name="test", user_id="user-2", session_id="sess-2")

        result = await service.list_sessions(app_name="test")
        assert len(result.sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_filtered_by_user(self, tmp_path: object) -> None:
        """list_sessions filters by user_id when provided."""
        service = CheckpointSessionService(str(tmp_path))
        await service.create_session(app_name="test", user_id="user-1", session_id="sess-1")
        await service.create_session(app_name="test", user_id="user-2", session_id="sess-2")

        result = await service.list_sessions(app_name="test", user_id="user-1")
        assert len(result.sessions) == 1
        assert result.sessions[0].id == "sess-1"

    @pytest.mark.asyncio
    async def test_create_with_initial_state(self, tmp_path: object) -> None:
        """create_session accepts initial state."""
        service = CheckpointSessionService(str(tmp_path))
        session = await service.create_session(
            app_name="test",
            user_id="user-1",
            session_id="sess-1",
            state={"initial": "value"},
        )

        assert session.state["initial"] == "value"

    @pytest.mark.asyncio
    async def test_checkpoint_round_trip(self, tmp_path: object) -> None:
        """Session state survives checkpoint write + read."""
        service = CheckpointSessionService(str(tmp_path))
        session = await service.create_session(
            app_name="test",
            user_id="user-1",
            session_id="sess-rt",
            state={"count": 42},
        )

        # Append event with state change
        event = Event(
            invocation_id="inv-1",
            author="agent",
            actions=EventActions(state_delta={"extra": "data"}),
        )
        await service.append_event(session, event)

        # Simulate restart — clear memory, reload from checkpoint
        service._sessions.clear()
        loaded = await service.get_session(app_name="test", user_id="user-1", session_id="sess-rt")

        assert loaded is not None
        assert loaded.state.get("count") == 42
        # Note: state_delta applied to session.state before checkpoint write,
        # so "extra" should be in the checkpoint
        assert loaded.state.get("extra") == "data"
