"""Tests for TeammateSessionManager."""

from __future__ import annotations

import asyncio
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


class TestHistorySync:
    def test_workspace_params_stored(self) -> None:
        mock_ws = MagicMock()
        t = _make_teammate(workspace_client=mock_ws, workspace_id="ws-1", session_id="s-1")
        assert t._workspace_client is mock_ws
        assert t._workspace_id == "ws-1"
        assert t._session_id == "s-1"

    def test_default_session_id(self) -> None:
        t = _make_teammate(name="analyst")
        assert t._session_id == "teammate-analyst"

    def test_sync_interval_default(self) -> None:
        t = _make_teammate()
        assert t._sync_interval == 5

    async def test_on_step_complete_syncs_at_interval(self) -> None:
        mock_ws = MagicMock()
        mock_ws.upload_session_history = AsyncMock()
        t = _make_teammate(
            workspace_client=mock_ws,
            workspace_id="ws-1",
            sync_interval=3,
        )

        # Steps 1 and 2 should not trigger sync (interval=3)
        await t._on_step_complete("task-1", 1)
        await t._on_step_complete("task-1", 2)
        mock_ws.upload_session_history.assert_not_called()

        # Step 3 should trigger sync (3 - 0 >= 3)
        await t._on_step_complete("task-1", 3)
        mock_ws.upload_session_history.assert_called_once()
        assert t._last_sync_step == 3

    async def test_on_step_complete_noop_without_workspace(self) -> None:
        t = _make_teammate()  # no workspace_client
        await t._on_step_complete("task-1", 5)  # should not raise

    async def test_on_step_complete_noop_with_zero_interval(self) -> None:
        mock_ws = MagicMock()
        mock_ws.upload_session_history = AsyncMock()
        t = _make_teammate(
            workspace_client=mock_ws,
            workspace_id="ws-1",
            sync_interval=0,
        )
        await t._on_step_complete("task-1", 5)
        mock_ws.upload_session_history.assert_not_called()

    async def test_sync_history_failure_is_best_effort(self) -> None:
        mock_ws = MagicMock()
        mock_ws.upload_session_history = AsyncMock(side_effect=RuntimeError("network"))
        t = _make_teammate(workspace_client=mock_ws, workspace_id="ws-1")
        # Add a message so there's something to sync
        t._thread.add_user_message("hello")
        await t._sync_history("task-1")  # should not raise


class TestTeamId:
    def test_team_id_stored(self) -> None:
        t = _make_teammate(team_id="tm-123")
        assert t._team_id == "tm-123"

    def test_team_id_defaults_to_empty(self) -> None:
        t = _make_teammate()
        assert t._team_id == ""


class TestTeammateEventProxy:
    def test_emit_text_chunk_emits_teammate_output_only(self) -> None:
        from agent_host.teams.teammate_session import _TeammateEventProxy

        delegate = MagicMock()
        proxy = _TeammateEventProxy(delegate, team_id="tm-1", teammate_name="researcher")
        proxy.emit_text_chunk("task-1", "hello world", step_id="s-1")

        # Should NOT forward text_chunk to lead's conversation
        delegate.emit_text_chunk.assert_not_called()
        # Should emit teammate_output for the team UI
        delegate.emit_teammate_output.assert_called_once_with("tm-1", "researcher", "hello world")

    def test_tool_requested_emits_teammate_tool_only(self) -> None:
        from agent_host.teams.teammate_session import _TeammateEventProxy

        delegate = MagicMock()
        proxy = _TeammateEventProxy(delegate, team_id="tm-1", teammate_name="worker")
        proxy.emit_tool_requested("ReadFile", "File.Read", {"path": "/a"}, tool_call_id="tc-1")

        delegate.emit_tool_requested.assert_not_called()
        delegate.emit_teammate_tool.assert_called_once_with(
            "tm-1", "worker", "ReadFile", "requested", "tc-1", args="/a"
        )

    def test_tool_completed_emits_teammate_tool_only(self) -> None:
        from agent_host.teams.teammate_session import _TeammateEventProxy

        delegate = MagicMock()
        proxy = _TeammateEventProxy(delegate, team_id="tm-1", teammate_name="worker")
        proxy.emit_tool_completed("ReadFile", "success", tool_call_id="tc-1")

        delegate.emit_tool_completed.assert_not_called()
        delegate.emit_teammate_tool.assert_called_once_with(
            "tm-1", "worker", "ReadFile", "success", "tc-1"
        )

    def test_session_events_suppressed(self) -> None:
        from agent_host.teams.teammate_session import _TeammateEventProxy

        delegate = MagicMock()
        proxy = _TeammateEventProxy(delegate, team_id="tm-1", teammate_name="worker")
        # Session events should be no-ops
        proxy.emit_step_started("task-1", 1)
        proxy.emit_step_completed("task-1", 1)
        delegate.emit_step_started.assert_not_called()
        delegate.emit_step_completed.assert_not_called()

    def test_team_notification_methods_forwarded(self) -> None:
        from agent_host.teams.teammate_session import _TeammateEventProxy

        delegate = MagicMock()
        proxy = _TeammateEventProxy(delegate, team_id="tm-1", teammate_name="worker")
        proxy.emit_team_created("tm-1", "my-team")
        delegate.emit_team_created.assert_called_once_with("tm-1", "my-team")


class TestOnActivityCallback:
    """Verify on_activity callback is invoked on each step."""

    async def test_on_step_complete_calls_activity_callback(self) -> None:
        calls: list[str] = []
        t = _make_teammate(on_activity=lambda name: calls.append(name))
        await t._on_step_complete("task-1", 1)
        assert calls == ["worker"]

    async def test_on_step_complete_no_callback_is_safe(self) -> None:
        t = _make_teammate()  # no on_activity
        await t._on_step_complete("task-1", 1)  # should not raise


class TestExitCheck:
    """Verify teammate exit check prevents premature termination.

    The exit check uses team-wide incomplete task detection (not filtered
    by assignee/created_by) since tasks are pulled, not pushed.
    """

    async def test_no_task_list_allows_exit(self) -> None:
        t = _make_teammate()  # no task_list
        result = await t._check_incomplete_tasks()
        assert result is None  # exit allowed

    async def test_no_incomplete_tasks_allows_exit(self) -> None:
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        task = await tl.create_task("Done task", "desc", created_by="worker")
        await tl.update_status(task.task_id, "completed", result="done")
        t = _make_teammate(task_list=tl)
        result = await t._check_incomplete_tasks()
        assert result is None  # exit allowed

    async def test_incomplete_task_blocks_on_wake_event(self) -> None:
        """Any incomplete team task causes the exit check to block on wake_event."""
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        await tl.create_task("Team work", "do stuff", created_by="lead")
        wake = asyncio.Event()
        t = _make_teammate(task_list=tl, wake_event=wake)

        # Should block until wake_event is set, then return nudge
        async def wake_later() -> None:
            await asyncio.sleep(0.05)
            wake.set()

        bg_task = asyncio.create_task(wake_later())  # noqa: F841, RUF006
        result = await t._check_incomplete_tasks()
        assert result is not None
        assert "1 incomplete task" in result
        assert "Team work" in result

    async def test_incomplete_task_without_wake_event_returns_nudge(self) -> None:
        """Without a wake_event, exit check returns nudge immediately."""
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        await tl.create_task("Team work", "do stuff", created_by="lead")
        t = _make_teammate(task_list=tl)  # no wake_event
        result = await t._check_incomplete_tasks()
        assert result is not None
        assert "1 incomplete task" in result

    async def test_blocked_task_waits_on_wake_event(self) -> None:
        """When tasks are blocked, exit check blocks then returns nudge."""
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        t1 = await tl.create_task("Prereq", "research", created_by="researcher")
        await tl.create_task(
            "My report", "write report", created_by="lead", blocked_by=[t1.task_id]
        )
        wake = asyncio.Event()
        t = _make_teammate(task_list=tl, wake_event=wake)

        # Unblock the task after a short delay
        async def unblock_later() -> None:
            await asyncio.sleep(0.05)
            await tl.update_status(t1.task_id, "completed")
            wake.set()

        bg_task = asyncio.create_task(unblock_later())  # noqa: F841, RUF006
        result = await t._check_incomplete_tasks()
        # After unblocking, the task is pending so we still get a nudge
        assert result is not None
        assert "pending" in result

    async def test_all_tasks_completed_allows_exit(self) -> None:
        """If all tasks complete while waiting, exit is allowed."""
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        t1 = await tl.create_task("Prereq", "research", created_by="researcher")
        t2 = await tl.create_task(
            "Report", "write report", created_by="lead", blocked_by=[t1.task_id]
        )
        wake = asyncio.Event()
        t = _make_teammate(task_list=tl, wake_event=wake)

        # Complete both tasks before wake
        async def complete_all() -> None:
            await asyncio.sleep(0.05)
            await tl.update_status(t1.task_id, "completed")
            await tl.update_status(t2.task_id, "completed", result="done")
            wake.set()

        bg_task = asyncio.create_task(complete_all())  # noqa: F841, RUF006
        result = await t._check_incomplete_tasks()
        assert result is None  # exit allowed

    async def test_any_team_task_blocks_exit(self) -> None:
        """Any incomplete team task blocks exit — not filtered by owner."""
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        await tl.create_task("Other's work", "something", created_by="other_agent")
        wake = asyncio.Event()
        t = _make_teammate(task_list=tl, wake_event=wake)

        async def wake_later() -> None:
            await asyncio.sleep(0.05)
            wake.set()

        bg_task = asyncio.create_task(wake_later())  # noqa: F841, RUF006
        result = await t._check_incomplete_tasks()
        assert result is not None  # exit blocked — team has incomplete work


class TestWaitForWork:
    """Verify deterministic pre-LLM-call blocking when no work is available."""

    async def test_no_task_list_returns_immediately(self) -> None:
        t = _make_teammate()  # no task_list
        await t._wait_for_work()  # should not block

    async def test_pending_task_returns_immediately(self) -> None:
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        await tl.create_task("Available work", "desc")
        t = _make_teammate(task_list=tl)
        await t._wait_for_work()  # should return immediately

    async def test_in_progress_task_returns_immediately(self) -> None:
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        task = await tl.create_task("Active work", "desc")
        await tl.update_status(task.task_id, "in_progress")
        t = _make_teammate(task_list=tl)
        await t._wait_for_work()  # should return immediately

    async def test_all_tasks_done_returns_immediately(self) -> None:
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        task = await tl.create_task("Done", "desc")
        await tl.update_status(task.task_id, "completed")
        t = _make_teammate(task_list=tl)
        await t._wait_for_work()  # should return — team is done

    async def test_blocked_only_blocks_until_wake(self) -> None:
        """When only blocked tasks exist, blocks on wake_event."""
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        t1 = await tl.create_task("Prereq", "research")
        await tl.create_task("Blocked", "needs prereq", blocked_by=[t1.task_id])
        wake = asyncio.Event()
        t = _make_teammate(task_list=tl, wake_event=wake)

        # Complete the prereq after a delay, which unblocks the dependent task
        async def unblock_later() -> None:
            await asyncio.sleep(0.05)
            await tl.update_status(t1.task_id, "completed")
            wake.set()

        bg_task = asyncio.create_task(unblock_later())  # noqa: F841, RUF006
        await t._wait_for_work()  # should block then return when pending task appears

    async def test_no_wake_event_returns_immediately(self) -> None:
        """Without wake_event, cannot block — returns immediately."""
        from agent_host.teams.task_list import SharedTaskList

        tl = SharedTaskList()
        t1 = await tl.create_task("Prereq", "research")
        await tl.create_task("Blocked", "needs prereq", blocked_by=[t1.task_id])
        t = _make_teammate(task_list=tl)  # no wake_event
        await t._wait_for_work()  # should return immediately
