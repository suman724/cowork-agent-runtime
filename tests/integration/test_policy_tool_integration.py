"""Integration tests — PolicyEnforcer + ToolAdapter + ApprovalGate in realistic flows."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from cowork_platform.tool_definition import ToolDefinition
from cowork_platform.tool_result import ToolResult

from agent_host.agent.tool_adapter import ToolExecutionDeps, adapt_tools
from agent_host.approval.approval_gate import ApprovalGate
from agent_host.policy.policy_enforcer import PolicyEnforcer
from tests.fixtures.policy_bundles import make_policy_bundle, make_restrictive_bundle
from tool_runtime.models import ToolExecutionResult


def _make_router(
    tools: list[str] | None = None,
) -> MagicMock:
    """Create a mock ToolRouter with specified tools."""
    if tools is None:
        tools = ["ReadFile", "WriteFile", "DeleteFile"]
    router = MagicMock()
    router.get_available_tools.return_value = [
        ToolDefinition(
            toolName=name,
            description=f"{name} tool",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        for name in tools
    ]
    return router


def _make_success_result(tool_name: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_result=ToolResult(
            toolName=tool_name,
            sessionId="sess-1",
            taskId="task-1",
            stepId="step-1",
            status="succeeded",
            outputText=f"{tool_name} succeeded",
        ),
    )


@pytest.mark.integration
class TestPolicyToolIntegration:
    @pytest.mark.asyncio
    async def test_denied_tool_blocked(self) -> None:
        """File.Write with blocked path → tool returns CAPABILITY_DENIED via before_callback."""
        from agent_host.agent.callbacks import make_before_tool_callback

        bundle = make_restrictive_bundle(allowed_paths=["/allowed"])
        enforcer = PolicyEnforcer(bundle)
        callback = make_before_tool_callback(enforcer)

        result = callback(MagicMock(), "WriteFile", {"path": "/forbidden/secret.txt"})
        assert result is not None
        assert result["status"] == "denied"
        assert result["error"]["code"] == "CAPABILITY_DENIED"

    @pytest.mark.asyncio
    async def test_allowed_tool_executes(self) -> None:
        """File.Read with allowed path → tool executes successfully."""
        router = _make_router(["ReadFile"])
        router.execute = AsyncMock(return_value=_make_success_result("ReadFile"))

        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)

        tools = adapt_tools(router, enforcer, session_id="sess-1", task_id="task-1")
        assert len(tools) == 1

        fn = tools[0].func
        result = await fn(path="/tmp/allowed.txt")
        assert result["status"] == "succeeded"
        assert result["output"] == "ReadFile succeeded"

    @pytest.mark.asyncio
    async def test_approval_required_waits_then_approved(self) -> None:
        """Tool with requiresApproval → tool_fn awaits gate → deliver approved → executes."""
        router = _make_router(["WriteFile"])
        router.execute = AsyncMock(return_value=_make_success_result("WriteFile"))

        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        gate = ApprovalGate()
        emitter = MagicMock()

        deps = ToolExecutionDeps(
            tool_router=router,
            policy_enforcer=enforcer,
            event_emitter=emitter,
            approval_gate=gate,
            approval_timeout=5.0,
            session_id="sess-1",
            task_id="task-1",
        )

        tools = adapt_tools(router, enforcer, deps=deps)
        fn = tools[0].func

        async def _approve_soon() -> None:
            await asyncio.sleep(0.01)
            call_kwargs = emitter.emit_approval_requested.call_args[1]
            gate.deliver(call_kwargs["approval_id"], "approved")

        _task = asyncio.create_task(_approve_soon())  # noqa: RUF006
        result = await fn(path="/tmp/file.txt", content="data")

        assert result["status"] == "succeeded"
        router.execute.assert_called_once()
        emitter.emit_approval_requested.assert_called_once()

    @pytest.mark.asyncio
    async def test_approval_denied_skips_tool(self) -> None:
        """Deliver denied → tool returns APPROVAL_DENIED without executing."""
        router = _make_router(["WriteFile"])
        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        gate = ApprovalGate()
        emitter = MagicMock()

        deps = ToolExecutionDeps(
            tool_router=router,
            policy_enforcer=enforcer,
            event_emitter=emitter,
            approval_gate=gate,
            approval_timeout=5.0,
            session_id="sess-1",
            task_id="task-1",
        )

        tools = adapt_tools(router, enforcer, deps=deps)
        fn = tools[0].func

        async def _deny_soon() -> None:
            await asyncio.sleep(0.01)
            call_kwargs = emitter.emit_approval_requested.call_args[1]
            gate.deliver(call_kwargs["approval_id"], "denied")

        _task = asyncio.create_task(_deny_soon())  # noqa: RUF006
        result = await fn(path="/tmp/file.txt", content="data")

        assert result["status"] == "denied"
        assert result["error"]["code"] == "APPROVAL_DENIED"
        router.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_approval_timeout(self) -> None:
        """No delivery → times out → denied (short 0.05s timeout)."""
        router = _make_router(["WriteFile"])
        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        gate = ApprovalGate()
        emitter = MagicMock()

        deps = ToolExecutionDeps(
            tool_router=router,
            policy_enforcer=enforcer,
            event_emitter=emitter,
            approval_gate=gate,
            approval_timeout=0.05,
            session_id="sess-1",
            task_id="task-1",
        )

        tools = adapt_tools(router, enforcer, deps=deps)
        fn = tools[0].func

        result = await fn(path="/tmp/file.txt", content="data")

        assert result["status"] == "denied"
        assert result["error"]["code"] == "APPROVAL_DENIED"
        router.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_expired_policy_denies_all(self) -> None:
        """Expired policy → all tools denied via before_tool_callback."""
        from agent_host.agent.callbacks import make_before_tool_callback

        bundle = make_policy_bundle(expired=True)
        enforcer = PolicyEnforcer(bundle)
        callback = make_before_tool_callback(enforcer)

        # All known tools should be denied
        for tool_name in ["ReadFile", "WriteFile", "DeleteFile", "RunCommand", "HttpRequest"]:
            result = callback(MagicMock(), tool_name, {"path": "/tmp/x"})
            assert result is not None, f"{tool_name} should be denied"
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_before_callback_blocks_denied(self) -> None:
        """before_tool_callback returns block dict for DENIED decision."""
        from agent_host.agent.callbacks import make_before_tool_callback

        # Bundle only allows LLM.Call — File.Read is not granted
        bundle = make_policy_bundle(capabilities=[{"name": "LLM.Call"}])
        enforcer = PolicyEnforcer(bundle)
        callback = make_before_tool_callback(enforcer)

        result = callback(MagicMock(), "ReadFile", {"path": "/tmp/test"})
        assert result is not None
        assert result["status"] == "denied"
        assert result["error"]["code"] == "CAPABILITY_DENIED"
        assert "not granted" in result["error"]["message"].lower() or result["error"]["message"]
