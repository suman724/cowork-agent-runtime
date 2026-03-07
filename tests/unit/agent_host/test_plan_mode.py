"""Tests for plan mode — tool filtering, enter/exit, locked mode."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from agent_host.llm.models import ToolCallMessage
from agent_host.loop.agent_tools import AgentToolHandler
from agent_host.loop.tool_executor import ToolExecutor
from agent_host.memory.working_memory import WorkingMemory
from tests.fixtures.policy_bundles import make_policy_bundle


def _make_tool_router_mock() -> MagicMock:
    router = MagicMock()
    router.get_available_tools.return_value = [
        MagicMock(toolName="ReadFile", description="Read", inputSchema={}),
        MagicMock(toolName="WriteFile", description="Write", inputSchema={}),
        MagicMock(toolName="RunCommand", description="Run", inputSchema={}),
        MagicMock(toolName="GrepFiles", description="Grep", inputSchema={}),
    ]

    tool_result = MagicMock()
    tool_result.status = "succeeded"
    tool_result.outputText = "done"
    tool_result.error = None

    exec_result = MagicMock()
    exec_result.tool_result = tool_result
    exec_result.artifacts = []
    exec_result.image_content = None

    router.execute = AsyncMock(return_value=exec_result)
    return router


class TestPlanModeToolFiltering:
    def test_plan_mode_filters_tool_definitions(self) -> None:
        """In plan mode, only read-only tools should be returned."""
        from agent_host.policy.policy_enforcer import PolicyEnforcer

        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            plan_mode=True,
        )

        defs = executor.get_tool_definitions()
        names = {d["function"]["name"] for d in defs}
        assert "ReadFile" in names
        assert "GrepFiles" in names
        assert "WriteFile" not in names
        assert "RunCommand" not in names

    def test_normal_mode_returns_all_tools(self) -> None:
        """Without plan mode, all tools should be returned."""
        from agent_host.policy.policy_enforcer import PolicyEnforcer

        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            plan_mode=False,
        )

        defs = executor.get_tool_definitions()
        names = {d["function"]["name"] for d in defs}
        assert "WriteFile" in names
        assert "RunCommand" in names


class TestPlanModeEnforcement:
    async def test_blocked_tool_returns_denial(self) -> None:
        """Calling a write tool in plan mode should return PLAN_MODE_RESTRICTED."""
        from agent_host.policy.policy_enforcer import PolicyEnforcer

        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            plan_mode=True,
            session_id="sess-1",
        )

        call = ToolCallMessage(id="tc1", name="WriteFile", arguments={"path": "/a"})
        results = await executor.execute_tool_calls([call], "task-1")

        assert results[0].status == "denied"
        data = json.loads(results[0].result_text)
        assert data["error"]["code"] == "PLAN_MODE_RESTRICTED"
        router.execute.assert_not_awaited()

    async def test_allowed_tool_executes_in_plan_mode(self) -> None:
        """Calling a read tool in plan mode should succeed."""
        from agent_host.policy.policy_enforcer import PolicyEnforcer

        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            plan_mode=True,
            session_id="sess-1",
        )

        call = ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/a"})
        results = await executor.execute_tool_calls([call], "task-1")
        assert results[0].status == "succeeded"


class TestPlanModeProperty:
    def test_plan_mode_setter_when_not_locked(self) -> None:
        """plan_mode setter should work when not locked."""
        from agent_host.policy.policy_enforcer import PolicyEnforcer

        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        executor = ToolExecutor(tool_router=router, policy_enforcer=enforcer, plan_mode=False)
        executor.plan_mode = True
        assert executor.plan_mode is True

    def test_plan_mode_setter_ignored_when_locked(self) -> None:
        """plan_mode setter should be no-op when locked."""
        from agent_host.policy.policy_enforcer import PolicyEnforcer

        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            plan_mode=True,
            plan_mode_locked=True,
        )
        executor.plan_mode = False
        assert executor.plan_mode is True  # still locked


class TestEnterExitPlanMode:
    def test_enter_plan_mode(self) -> None:
        """EnterPlanMode should set plan_mode=True."""
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        assert handler._plan_mode is False

        result = handler._handle_enter_plan_mode()
        assert result["status"] == "success"
        assert result["planMode"] is True
        assert handler._plan_mode is True

    def test_exit_plan_mode(self) -> None:
        """ExitPlanMode should set plan_mode=False."""
        wm = WorkingMemory()
        handler = AgentToolHandler(wm, plan_mode=True)

        result = handler._handle_exit_plan_mode()
        assert result["status"] == "success"
        assert result["planMode"] is False
        assert handler._plan_mode is False

    def test_enter_already_in_plan_mode(self) -> None:
        """EnterPlanMode when already in plan mode should be noop."""
        wm = WorkingMemory()
        handler = AgentToolHandler(wm, plan_mode=True)

        result = handler._handle_enter_plan_mode()
        assert result["status"] == "noop"

    def test_exit_when_locked(self) -> None:
        """ExitPlanMode with plan_mode_locked should fail."""
        wm = WorkingMemory()
        handler = AgentToolHandler(wm, plan_mode=True, plan_mode_locked=True)

        result = handler._handle_exit_plan_mode()
        assert result["status"] == "error"
        assert handler._plan_mode is True

    def test_callback_called_on_enter(self) -> None:
        """on_plan_mode_changed should be called when entering plan mode."""
        callbacks: list[tuple[bool, str]] = []

        def on_change(mode: bool, source: str) -> None:
            callbacks.append((mode, source))

        wm = WorkingMemory()
        handler = AgentToolHandler(wm, on_plan_mode_changed=on_change)
        handler._handle_enter_plan_mode()

        assert callbacks == [(True, "agent")]

    def test_callback_called_on_exit(self) -> None:
        """on_plan_mode_changed should be called when exiting plan mode."""
        callbacks: list[tuple[bool, str]] = []

        def on_change(mode: bool, source: str) -> None:
            callbacks.append((mode, source))

        wm = WorkingMemory()
        handler = AgentToolHandler(wm, plan_mode=True, on_plan_mode_changed=on_change)
        handler._handle_exit_plan_mode()

        assert callbacks == [(False, "agent")]

    def test_plan_mode_tool_definitions_present(self) -> None:
        """EnterPlanMode and ExitPlanMode should be in tool definitions."""
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        defs = handler.get_tool_definitions()
        names = {d["function"]["name"] for d in defs}
        assert "EnterPlanMode" in names
        assert "ExitPlanMode" in names
