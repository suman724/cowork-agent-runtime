"""Tests for team data models."""

from __future__ import annotations

from agent_host.teams.models import (
    TeamConfig,
    TeammateInfo,
    TeamMessage,
    TeamTask,
)


class TestTeamConfig:
    def test_defaults(self) -> None:
        cfg = TeamConfig()
        assert cfg.name == "default"
        assert cfg.max_teammates == 5

    def test_custom_values(self) -> None:
        cfg = TeamConfig(name="research", description="Research team", max_teammates=3)
        assert cfg.name == "research"
        assert cfg.max_teammates == 3


class TestTeammateInfo:
    def test_defaults(self) -> None:
        info = TeammateInfo(name="worker", role="coder", budget=1000)
        assert info.status == "created"
        assert info.session_id is None

    def test_custom_status(self) -> None:
        info = TeammateInfo(name="worker", role="coder", budget=1000, status="running")
        assert info.status == "running"


class TestTeamTask:
    def test_auto_id_and_timestamps(self) -> None:
        task = TeamTask(title="Test task", description="Do something")
        assert len(task.task_id) == 12
        assert task.status == "pending"
        assert task.created_at is not None
        assert task.blocked_by == []
        assert task.result is None

    def test_unique_ids(self) -> None:
        t1 = TeamTask(title="a")
        t2 = TeamTask(title="b")
        assert t1.task_id != t2.task_id


class TestTeamMessage:
    def test_defaults(self) -> None:
        msg = TeamMessage(from_agent="lead", content="hello")
        assert msg.to_agent is None
        assert msg.message_type == "message"
        assert len(msg.message_id) == 12

    def test_broadcast_type(self) -> None:
        msg = TeamMessage(from_agent="lead", content="hi", message_type="broadcast")
        assert msg.message_type == "broadcast"
