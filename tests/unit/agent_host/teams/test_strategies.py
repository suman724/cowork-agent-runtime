"""Tests for team strategy implementations."""

from __future__ import annotations

import pytest

from agent_host.coordination.protocols import (
    CheckpointStrategy,
    ContextInjectionStrategy,
    CoordinationStrategy,
    ToolProviderStrategy,
)
from agent_host.teams.strategies import (
    TeamCheckpointProvider,
    TeamContextInjector,
    TeamCoordinator,
    TeamToolProvider,
)

# ── Protocol conformance ──────────────────────────────────────────


class TestProtocolConformance:
    def test_coordinator_satisfies_protocol(self) -> None:
        assert isinstance(TeamCoordinator(), CoordinationStrategy)

    def test_tool_provider_satisfies_protocol(self) -> None:
        assert isinstance(TeamToolProvider(TeamCoordinator()), ToolProviderStrategy)

    def test_context_injector_satisfies_protocol(self) -> None:
        assert isinstance(TeamContextInjector(TeamCoordinator()), ContextInjectionStrategy)

    def test_checkpoint_provider_satisfies_protocol(self) -> None:
        assert isinstance(TeamCheckpointProvider(TeamCoordinator()), CheckpointStrategy)


# ── TeamCoordinator ──────────────────────────────────────────────


class TestTeamCoordinator:
    async def test_on_session_start_stores_session_id(self) -> None:
        coord = TeamCoordinator()
        await coord.on_session_start("sess-1", {})
        assert coord._lead_session_id == "sess-1"

    async def test_create_team(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        manager = coord.create_team("research", "Find data")
        assert manager is not None
        assert coord.is_team_active

    async def test_create_team_twice_raises(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        with pytest.raises(ValueError, match="already active"):
            coord.create_team("t2")

    async def test_spawn_agent_without_team_raises(self) -> None:
        coord = TeamCoordinator()
        with pytest.raises(RuntimeError, match="No active team"):
            await coord.spawn_agent("w", "coder", "do stuff", 1000)

    async def test_spawn_agent_with_team(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        result = await coord.spawn_agent("researcher", "research", "find data", 5000)
        assert result["name"] == "researcher"
        assert result["initial_prompt"] == "find data"

    async def test_get_active_agents(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        await coord.spawn_agent("w1", "coder", "code", 1000)
        await coord.spawn_agent("w2", "tester", "test", 2000)
        agents = coord.get_active_agents()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert names == {"w1", "w2"}

    async def test_shutdown_agent(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        await coord.spawn_agent("w1", "coder", "code", 1000)
        await coord.shutdown_agent("w1")
        assert coord.get_active_agents() == []

    async def test_on_session_shutdown_cleans_up(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        await coord.spawn_agent("w1", "coder", "code", 1000)
        await coord.on_session_shutdown()
        assert not coord.is_team_active

    async def test_shutdown_without_team_is_noop(self) -> None:
        coord = TeamCoordinator()
        await coord.on_session_shutdown()  # should not raise


# ── TeamToolProvider ──────────────────────────────────────────────


class TestTeamToolProvider:
    def test_lead_gets_create_team_before_team_active(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        tp = TeamToolProvider(coord)
        defs = tp.get_tool_definitions("lead")
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "CreateTeam"

    def test_solo_gets_nothing(self) -> None:
        coord = TeamCoordinator()
        tp = TeamToolProvider(coord)
        assert tp.get_tool_definitions("solo") == []

    def test_lead_gets_all_tools_when_team_active(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        tp = TeamToolProvider(coord)
        defs = tp.get_tool_definitions("lead")
        names = {d["function"]["name"] for d in defs}
        assert "CreateTeam" in names
        assert "CreateTeammate" in names
        assert "TeamTaskCreate" in names
        assert "SendTeamMessage" in names
        assert len(defs) == 8

    def test_teammate_gets_shared_tools_only(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        tp = TeamToolProvider(coord)
        defs = tp.get_tool_definitions("teammate")
        names = {d["function"]["name"] for d in defs}
        assert "TeamTaskCreate" in names
        assert "SendTeamMessage" in names
        assert "CreateTeam" not in names
        assert "CreateTeammate" not in names
        assert len(defs) == 4

    def test_owns_tool(self) -> None:
        coord = TeamCoordinator()
        tp = TeamToolProvider(coord)
        assert tp.owns_tool("CreateTeam") is True
        assert tp.owns_tool("TeamTaskCreate") is True
        assert tp.owns_tool("ReadFile") is False

    async def test_handle_create_team(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        tp = TeamToolProvider(coord)
        result = await tp.handle_tool_call("CreateTeam", {"name": "myteam"}, "lead")
        assert result["status"] == "success"
        assert coord.is_team_active

    async def test_handle_create_teammate(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        tp = TeamToolProvider(coord)
        result = await tp.handle_tool_call(
            "CreateTeammate",
            {"name": "worker", "role": "coder", "initial_prompt": "write code"},
            "lead",
        )
        assert result["status"] == "success"
        assert result["name"] == "worker"

    async def test_handle_task_create(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        tp = TeamToolProvider(coord)
        result = await tp.handle_tool_call(
            "TeamTaskCreate",
            {"title": "Build API", "description": "REST endpoints"},
            "lead",
        )
        assert result["status"] == "success"
        assert "task_id" in result

    async def test_handle_task_list(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        tp = TeamToolProvider(coord)
        await tp.handle_tool_call("TeamTaskCreate", {"title": "Task A", "description": "a"}, "lead")
        result = await tp.handle_tool_call("TeamTaskList", {}, "lead")
        assert result["status"] == "success"
        assert len(result["tasks"]) == 1

    async def test_handle_send_message(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        await coord.spawn_agent("worker", "coder", "code", 1000)
        tp = TeamToolProvider(coord)
        result = await tp.handle_tool_call(
            "SendTeamMessage", {"to": "worker", "content": "hello"}, "lead"
        )
        assert result["status"] == "success"

    async def test_handle_broadcast(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        await coord.spawn_agent("w1", "coder", "code", 1000)
        await coord.spawn_agent("w2", "coder", "code", 1000)
        tp = TeamToolProvider(coord)
        result = await tp.handle_tool_call(
            "SendTeamMessage", {"to": "all", "content": "update"}, "lead"
        )
        assert result["status"] == "success"

    async def test_handle_unknown_tool_raises(self) -> None:
        coord = TeamCoordinator()
        tp = TeamToolProvider(coord)
        with pytest.raises(RuntimeError, match="does not handle"):
            await tp.handle_tool_call("UnknownTool", {}, "lead")

    async def test_handle_shutdown_teammate(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        await coord.spawn_agent("worker", "coder", "code", 1000)
        tp = TeamToolProvider(coord)
        result = await tp.handle_tool_call("ShutdownTeammate", {"name": "worker"}, "lead")
        assert result["status"] == "success"
        assert coord.get_active_agents() == []

    async def test_handle_shutdown_team(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        await coord.spawn_agent("w1", "coder", "code", 1000)
        tp = TeamToolProvider(coord)
        result = await tp.handle_tool_call("ShutdownTeam", {}, "lead")
        assert result["status"] == "success"


# ── TeamContextInjector ──────────────────────────────────────────


class TestTeamContextInjector:
    async def test_no_team_returns_empty(self) -> None:
        coord = TeamCoordinator()
        ci = TeamContextInjector(coord)
        assert await ci.get_injections("lead") == []

    async def test_no_messages_still_returns_task_summary(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        assert coord.manager is not None
        await coord.manager.task_list.create_task("Build API", "REST endpoints")
        ci = TeamContextInjector(coord)
        injections = await ci.get_injections("lead")
        assert len(injections) == 1
        assert "[Team Tasks]" in injections[0]
        assert "Build API" in injections[0]

    async def test_messages_injected(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        await coord.spawn_agent("worker", "coder", "code", 1000)
        assert coord.manager is not None
        await coord.manager.send_message("worker", "lead", "task done")
        ci = TeamContextInjector(coord)
        injections = await ci.get_injections("lead")
        msg_injection = [i for i in injections if "[Team Messages]" in i]
        assert len(msg_injection) == 1
        assert "@worker" in msg_injection[0]

    async def test_overhead_tokens_when_active(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        ci = TeamContextInjector(coord)
        assert ci.estimate_overhead_tokens() > 0

    async def test_overhead_tokens_when_inactive(self) -> None:
        coord = TeamCoordinator()
        ci = TeamContextInjector(coord)
        assert ci.estimate_overhead_tokens() == 0


# ── TeamCheckpointProvider ───────────────────────────────────────


class TestTeamCheckpointProvider:
    async def test_capture_empty_when_no_team(self) -> None:
        coord = TeamCoordinator()
        cp = TeamCheckpointProvider(coord)
        assert cp.capture() == {}

    async def test_capture_includes_team_state(self) -> None:
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("research")
        await coord.spawn_agent("worker", "coder", "code", 5000)
        assert coord.manager is not None
        await coord.manager.task_list.create_task("Build API", "REST endpoints", created_by="lead")

        cp = TeamCheckpointProvider(coord)
        state = cp.capture()
        assert state["config"]["name"] == "research"
        assert "worker" in state["members"]
        assert state["members"]["worker"]["role"] == "coder"
        assert len(state["tasks"]) == 1
        assert state["tasks"][0]["title"] == "Build API"

    async def test_restore_recreates_team(self) -> None:
        # Capture
        coord1 = TeamCoordinator(lead_session_id="sess-1")
        coord1.create_team("research")
        await coord1.spawn_agent("worker", "coder", "code", 5000)
        assert coord1.manager is not None
        await coord1.manager.task_list.create_task("Build API", "REST endpoints", created_by="lead")
        cp1 = TeamCheckpointProvider(coord1)
        state = cp1.capture()

        # Restore into a fresh coordinator
        coord2 = TeamCoordinator(lead_session_id="sess-1")
        cp2 = TeamCheckpointProvider(coord2)
        await cp2.restore(state)

        assert coord2.is_team_active
        assert coord2.manager is not None
        assert coord2.manager.config.name == "research"
        agents = coord2.get_active_agents()
        assert len(agents) == 1
        assert agents[0]["name"] == "worker"
        tasks = await coord2.manager.task_list.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == "Build API"

    async def test_restore_empty_is_noop(self) -> None:
        coord = TeamCoordinator()
        cp = TeamCheckpointProvider(coord)
        await cp.restore({})
        assert not coord.is_team_active
