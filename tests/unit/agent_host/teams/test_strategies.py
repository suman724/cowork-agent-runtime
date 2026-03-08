"""Tests for team strategy implementations."""

from __future__ import annotations

import unittest.mock
from unittest.mock import AsyncMock

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
        assert len(defs) == 9

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


# ── WaitForTeam ──────────────────────────────────────────────────


class TestWaitForTeam:
    async def test_wake_on_task_completed(self) -> None:
        """WaitForTeam should return when a task is marked completed."""
        import asyncio

        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("test-team")
        await coord.spawn_agent("worker", "coder", "code", 5000)
        assert coord.manager is not None
        task = await coord.manager.task_list.create_task("Do work", "desc", created_by="lead")

        tp = TeamToolProvider(coord)

        async def complete_task_after_delay() -> None:
            await asyncio.sleep(0.05)
            await tp.handle_tool_call(
                "TeamTaskUpdate",
                {"task_id": task.task_id, "status": "completed", "result": "done"},
                "worker",
            )

        bg = asyncio.create_task(complete_task_after_delay())
        result = await tp.handle_tool_call("WaitForTeam", {"timeout": 5}, "lead")
        await bg
        assert result["status"] == "success"
        assert result["wake_reason"] == "event"
        assert result["all_tasks_done"] is True

    async def test_wake_on_message_to_lead(self) -> None:
        """WaitForTeam should return when a message is sent to lead."""
        import asyncio

        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("test-team")
        await coord.spawn_agent("worker", "coder", "code", 5000)

        tp = TeamToolProvider(coord)

        async def send_msg_after_delay() -> None:
            await asyncio.sleep(0.05)
            await tp.handle_tool_call(
                "SendTeamMessage",
                {"to": "lead", "content": "Need help"},
                "worker",
            )

        bg = asyncio.create_task(send_msg_after_delay())
        result = await tp.handle_tool_call("WaitForTeam", {"timeout": 5}, "lead")
        await bg
        assert result["status"] == "success"
        assert result["wake_reason"] == "event"

    async def test_wake_on_broadcast(self) -> None:
        """WaitForTeam should return on broadcast messages."""
        import asyncio

        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("test-team")
        await coord.spawn_agent("worker", "coder", "code", 5000)

        tp = TeamToolProvider(coord)

        async def broadcast_after_delay() -> None:
            await asyncio.sleep(0.05)
            await tp.handle_tool_call(
                "SendTeamMessage",
                {"to": "all", "content": "Update"},
                "worker",
            )

        bg = asyncio.create_task(broadcast_after_delay())
        result = await tp.handle_tool_call("WaitForTeam", {"timeout": 5}, "lead")
        await bg
        assert result["status"] == "success"
        assert result["wake_reason"] == "event"

    async def test_timeout_returns_status(self) -> None:
        """WaitForTeam should return with timeout reason when nothing happens."""
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("test-team")

        tp = TeamToolProvider(coord)
        result = await tp.handle_tool_call("WaitForTeam", {"timeout": 0.1}, "lead")
        assert result["status"] == "success"
        assert result["wake_reason"] == "timeout"

    async def test_task_summary_in_result(self) -> None:
        """WaitForTeam should include task summary."""
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("test-team")
        assert coord.manager is not None
        t1 = await coord.manager.task_list.create_task("Task 1", "desc", created_by="lead")
        await coord.manager.task_list.create_task("Task 2", "desc", created_by="lead")
        await coord.manager.task_list.update_status(t1.task_id, "completed", result="done")

        # Pre-set wake so we don't block
        coord.wake()

        tp = TeamToolProvider(coord)
        result = await tp.handle_tool_call("WaitForTeam", {"timeout": 1}, "lead")
        assert result["task_summary"]["total"] == 2
        assert result["task_summary"]["completed"] == 1
        assert result["task_summary"]["pending"] == 1
        assert result["all_tasks_done"] is False

    async def test_wake_event_set_by_coordinator_wake(self) -> None:
        """Direct wake() call should unblock wait_for_wake."""
        import asyncio

        coord = TeamCoordinator()

        async def wake_after_delay() -> None:
            await asyncio.sleep(0.05)
            coord.wake()

        bg = asyncio.create_task(wake_after_delay())
        reason = await coord.wait_for_wake(timeout=5)
        await bg
        assert reason == "event"

    async def test_wait_for_wake_timeout(self) -> None:
        """wait_for_wake should return 'timeout' when no event fires."""
        coord = TeamCoordinator()
        reason = await coord.wait_for_wake(timeout=0.05)
        assert reason == "timeout"


# ── Resume Teammates (Crash Recovery) ────────────────────────────


class TestResumeTeammates:
    async def test_resume_skips_non_running_members(self) -> None:
        """Only members with status='running' should be re-spawned."""
        from unittest.mock import MagicMock

        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")

        # Set shared resources so teammates can be spawned
        coord.set_shared_resources(
            llm_client=MagicMock(),
            policy_enforcer=MagicMock(),
            tool_router=MagicMock(),
        )

        await coord.spawn_agent("w1", "coder", "code", 5000)
        await coord.spawn_agent("w2", "tester", "test", 5000)

        assert coord.manager is not None
        # Simulate w2 being stopped
        coord.manager.members["w2"].status = "stopped"

        # Cancel existing tasks and clear sessions to simulate crash state
        for task in coord._teammate_tasks.values():
            task.cancel()
        coord._teammate_sessions.clear()
        coord._teammate_tasks.clear()

        # Patch at the import source so the local import inside resume_teammates picks it up
        mock_session = MagicMock()
        mock_session.run = AsyncMock(return_value=None)
        mock_session.cancel = MagicMock()

        with unittest.mock.patch(
            "agent_host.teams.teammate_session.TeammateSessionManager",
            return_value=mock_session,
        ) as mock_cls:
            await coord.resume_teammates()
            # Only w1 (running) should be resumed, not w2 (stopped)
            assert mock_cls.call_count == 1
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["name"] == "w1"

        # Clean up
        for task in coord._teammate_tasks.values():
            task.cancel()

    async def test_resume_noop_without_team(self) -> None:
        """resume_teammates is a no-op when no team exists."""
        coord = TeamCoordinator()
        await coord.resume_teammates()  # should not raise

    async def test_resume_noop_without_shared_resources(self) -> None:
        """resume_teammates warns and skips when shared resources are missing."""
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        await coord.spawn_agent("w1", "coder", "code", 5000)
        # No set_shared_resources called
        await coord.resume_teammates()  # should not raise


# ── Team Summary Upload ──────────────────────────────────────────


class TestTeamSummaryUpload:
    async def test_upload_team_summary_on_shutdown(self) -> None:
        """on_session_shutdown should upload a team summary artifact."""
        from unittest.mock import AsyncMock, MagicMock

        mock_ws_client = MagicMock()
        mock_ws_client.upload_artifact = AsyncMock()

        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.set_shared_resources(
            llm_client=MagicMock(),
            policy_enforcer=MagicMock(),
            tool_router=MagicMock(),
            workspace_client=mock_ws_client,
            workspace_id="ws-1",
        )
        coord.create_team("research")
        assert coord.manager is not None
        await coord.manager.task_list.create_task("Task A", "desc", created_by="lead")

        await coord.on_session_shutdown()

        mock_ws_client.upload_artifact.assert_called_once()
        call_kwargs = mock_ws_client.upload_artifact.call_args[1]
        assert call_kwargs["workspace_id"] == "ws-1"
        assert call_kwargs["session_id"] == "sess-1"
        assert call_kwargs["artifact_type"] == "team_summary"
        assert call_kwargs["content_type"] == "application/json"

        import json

        summary = json.loads(call_kwargs["artifact_data"])
        assert summary["team_name"] == "research"
        assert len(summary["tasks"]) == 1
        assert summary["tasks"][0]["title"] == "Task A"

    async def test_upload_summary_skipped_without_workspace(self) -> None:
        """No upload when workspace_client or workspace_id is missing."""
        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.create_team("t1")
        # No workspace_client set
        await coord.on_session_shutdown()  # should not raise

    async def test_upload_summary_failure_is_best_effort(self) -> None:
        """Upload failure should not propagate."""
        from unittest.mock import AsyncMock, MagicMock

        mock_ws_client = MagicMock()
        mock_ws_client.upload_artifact = AsyncMock(side_effect=RuntimeError("network"))

        coord = TeamCoordinator(lead_session_id="sess-1")
        coord.set_shared_resources(
            llm_client=MagicMock(),
            policy_enforcer=MagicMock(),
            tool_router=MagicMock(),
            workspace_client=mock_ws_client,
            workspace_id="ws-1",
        )
        coord.create_team("t1")
        await coord.on_session_shutdown()  # should not raise


# ── Checkpoint Restore with Resume ────────────────────────────────


class TestCheckpointRestoreResume:
    async def test_restore_calls_resume_teammates(self) -> None:
        """restore() should call resume_teammates to re-spawn running loops."""
        import unittest.mock

        # Build state from a coordinator with a running member
        coord1 = TeamCoordinator(lead_session_id="sess-1")
        coord1.create_team("research")
        await coord1.spawn_agent("worker", "coder", "code", 5000)
        assert coord1.manager is not None
        cp1 = TeamCheckpointProvider(coord1)
        state = cp1.capture()

        # Restore into a fresh coordinator
        coord2 = TeamCoordinator(lead_session_id="sess-1")
        cp2 = TeamCheckpointProvider(coord2)

        patch = unittest.mock.patch.object(coord2, "resume_teammates", new_callable=AsyncMock)
        with patch as mock_resume:
            await cp2.restore(state)
            mock_resume.assert_called_once()
