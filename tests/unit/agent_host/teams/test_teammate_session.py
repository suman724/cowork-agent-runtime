"""Tests for TeammateSessionManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from agent_host.teams.strategies import TeamContextInjector, TeamCoordinator, TeamToolProvider
from agent_host.teams.teammate_session import TeammateSessionManager


def _make_teammate(**kwargs: object) -> TeammateSessionManager:
    """Create a TeammateSessionManager with mocked shared resources."""
    coord = TeamCoordinator(lead_session_id="sess-lead")
    coord.create_team("test-team")
    defaults: dict[str, object] = {
        "name": "worker",
        "role": "general coder",
        "team_name": "test-team",
        "llm_client": MagicMock(),
        "policy_enforcer": MagicMock(),
        "tool_router": MagicMock(),
        "workspace_dir": "/tmp/workspace",
        "tool_provider": TeamToolProvider(coord),
        "context_injector": TeamContextInjector(coord),
        "budget": 50_000,
    }
    defaults.update(kwargs)
    return TeammateSessionManager(**defaults)  # type: ignore[arg-type]


class TestConstruction:
    def test_creates_with_correct_role(self) -> None:
        t = _make_teammate(name="researcher", role="research expert")
        assert t.name == "researcher"
        assert t.role == "research expert"

    def test_system_prompt_contains_team_info(self) -> None:
        t = _make_teammate(name="analyst", role="data analyst", team_name="analytics")
        prompt = t.system_prompt
        assert "analytics" in prompt
        assert "analyst" in prompt
        assert "data analyst" in prompt

    def test_system_prompt_contains_workspace(self) -> None:
        t = _make_teammate(workspace_dir="/projects/myapp")
        assert "/projects/myapp" in t.system_prompt

    def test_fresh_token_budget(self) -> None:
        t = _make_teammate(budget=80_000)
        assert t._token_budget._max_session_tokens == 80_000

    def test_fresh_working_memory(self) -> None:
        t1 = _make_teammate(name="w1")
        t2 = _make_teammate(name="w2")
        assert t1._working_memory is not t2._working_memory

    def test_fresh_thread(self) -> None:
        t1 = _make_teammate(name="w1")
        t2 = _make_teammate(name="w2")
        assert t1._thread is not t2._thread


class TestCancel:
    def test_cancel_sets_event(self) -> None:
        t = _make_teammate()
        assert not t._cancel_event.is_set()
        t.cancel()
        assert t._cancel_event.is_set()

    def test_cancel_cancels_task(self) -> None:
        t = _make_teammate()
        mock_task = AsyncMock()
        mock_task.cancel = MagicMock()
        t._task = mock_task
        t.cancel()
        mock_task.cancel.assert_called_once()
