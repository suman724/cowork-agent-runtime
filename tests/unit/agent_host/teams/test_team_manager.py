"""Tests for TeamManager."""

from __future__ import annotations

import pytest

from agent_host.teams.models import TeamConfig
from agent_host.teams.team_manager import TeamManager


class TestInit:
    def test_creates_task_list_and_mailbox(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        assert tm.task_list is not None
        assert tm.mailbox is not None
        assert tm.members == {}

    def test_lead_registered_in_mailbox(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        assert not tm.mailbox.has_messages("lead")  # inbox exists, no error

    def test_team_id_generated(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        assert len(tm.team_id) == 12


class TestCreateTeammate:
    async def test_create_teammate(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        info = await tm.create_teammate("researcher", "research", budget=5000)
        assert info.name == "researcher"
        assert info.role == "research"
        assert info.budget == 5000
        assert info.status == "created"

    async def test_teammate_registered_in_mailbox(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        await tm.create_teammate("worker", "coder", budget=3000)
        # Can send to the teammate without error
        await tm.send_message("lead", "worker", "hello")
        assert tm.mailbox.has_messages("worker")

    async def test_duplicate_name_raises(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        await tm.create_teammate("worker", "coder", budget=3000)
        with pytest.raises(ValueError, match="already exists"):
            await tm.create_teammate("worker", "coder", budget=3000)

    async def test_reserved_name_lead_raises(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        with pytest.raises(ValueError, match="reserved name"):
            await tm.create_teammate("lead", "coder", budget=3000)

    async def test_max_teammates_enforced(self) -> None:
        cfg = TeamConfig(max_teammates=2)
        tm = TeamManager(lead_session_id="sess-1", config=cfg)
        await tm.create_teammate("w1", "coder", budget=1000)
        await tm.create_teammate("w2", "coder", budget=1000)
        with pytest.raises(ValueError, match="Team is full"):
            await tm.create_teammate("w3", "coder", budget=1000)


class TestShutdown:
    async def test_shutdown_teammate(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        await tm.create_teammate("worker", "coder", budget=3000)
        await tm.shutdown_teammate("worker")
        assert tm.get_teammate("worker") is None
        assert len(tm.get_active_agents()) == 0

    async def test_shutdown_nonexistent_raises(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        with pytest.raises(KeyError, match="not found"):
            await tm.shutdown_teammate("ghost")

    async def test_shutdown_all(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        await tm.create_teammate("w1", "coder", budget=1000)
        await tm.create_teammate("w2", "coder", budget=1000)
        await tm.shutdown_all()
        assert tm.get_active_agents() == []


class TestGetters:
    async def test_get_active_agents(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        await tm.create_teammate("w1", "coder", budget=1000)
        await tm.create_teammate("w2", "researcher", budget=2000)
        agents = tm.get_active_agents()
        assert len(agents) == 2
        names = {a.name for a in agents}
        assert names == {"w1", "w2"}

    async def test_get_teammate(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        await tm.create_teammate("worker", "coder", budget=3000)
        info = tm.get_teammate("worker")
        assert info is not None
        assert info.name == "worker"

    async def test_get_nonexistent_returns_none(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        assert tm.get_teammate("ghost") is None


class TestMessaging:
    async def test_send_message_delegates(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        await tm.create_teammate("worker", "coder", budget=3000)
        await tm.send_message("lead", "worker", "do this")
        messages = await tm.mailbox.poll("worker")
        assert len(messages) == 1
        assert messages[0].content == "do this"

    async def test_broadcast_delegates(self) -> None:
        tm = TeamManager(lead_session_id="sess-1")
        await tm.create_teammate("w1", "coder", budget=1000)
        await tm.create_teammate("w2", "coder", budget=1000)
        await tm.broadcast_message("lead", "team update")
        assert tm.mailbox.has_messages("w1")
        assert tm.mailbox.has_messages("w2")
        assert not tm.mailbox.has_messages("lead")
