"""Integration tests — PolicyEnforcer + ToolExecutor + ApprovalGate in realistic flows."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from cowork_platform.tool_definition import ToolDefinition
from cowork_platform.tool_result import ToolResult

from agent_host.approval.approval_gate import ApprovalGate
from agent_host.llm.models import ToolCallMessage
from agent_host.loop.tool_executor import ToolExecutor
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
        """File.Write with blocked path → tool returns CAPABILITY_DENIED."""
        bundle = make_restrictive_bundle(allowed_paths=["/allowed"])
        enforcer = PolicyEnforcer(bundle)
        router = _make_router(["WriteFile"])

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            session_id="sess-1",
        )

        call = ToolCallMessage(
            id="tc1", name="WriteFile", arguments={"path": "/forbidden/secret.txt"}
        )
        results = await executor.execute_tool_calls([call], "task-1")

        assert results[0].status == "denied"
        result_data = json.loads(results[0].result_text)
        assert result_data["error"]["code"] == "CAPABILITY_DENIED"

    @pytest.mark.asyncio
    async def test_allowed_tool_executes(self) -> None:
        """File.Read with allowed path → tool executes successfully."""
        router = _make_router(["ReadFile"])
        router.execute = AsyncMock(return_value=_make_success_result("ReadFile"))

        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            session_id="sess-1",
        )

        call = ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/tmp/allowed.txt"})
        results = await executor.execute_tool_calls([call], "task-1")

        assert results[0].status == "succeeded"
        router.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_approval_required_waits_then_approved(self) -> None:
        """Tool with requiresApproval → executor awaits gate → deliver approved → executes."""
        router = _make_router(["WriteFile"])
        router.execute = AsyncMock(return_value=_make_success_result("WriteFile"))

        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        gate = ApprovalGate()
        emitter = MagicMock()

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            event_emitter=emitter,
            approval_gate=gate,
            approval_timeout=5.0,
            session_id="sess-1",
        )

        call = ToolCallMessage(
            id="tc1", name="WriteFile", arguments={"path": "/tmp/file.txt", "content": "data"}
        )

        async def _approve_soon() -> None:
            await asyncio.sleep(0.01)
            call_kwargs = emitter.emit_approval_requested.call_args[1]
            gate.deliver(call_kwargs["approval_id"], "approved")

        _task = asyncio.create_task(_approve_soon())  # noqa: RUF006
        results = await executor.execute_tool_calls([call], "task-1")

        assert results[0].status == "succeeded"
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

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            event_emitter=emitter,
            approval_gate=gate,
            approval_timeout=5.0,
            session_id="sess-1",
        )

        call = ToolCallMessage(
            id="tc1", name="WriteFile", arguments={"path": "/tmp/file.txt", "content": "data"}
        )

        async def _deny_soon() -> None:
            await asyncio.sleep(0.01)
            call_kwargs = emitter.emit_approval_requested.call_args[1]
            gate.deliver(call_kwargs["approval_id"], "denied")

        _task = asyncio.create_task(_deny_soon())  # noqa: RUF006
        results = await executor.execute_tool_calls([call], "task-1")

        assert results[0].status == "denied"
        result_data = json.loads(results[0].result_text)
        assert result_data["error"]["code"] == "APPROVAL_DENIED"
        router.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_approval_timeout(self) -> None:
        """No delivery → times out → denied (short 0.05s timeout)."""
        router = _make_router(["WriteFile"])
        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        gate = ApprovalGate()
        emitter = MagicMock()

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            event_emitter=emitter,
            approval_gate=gate,
            approval_timeout=0.05,
            session_id="sess-1",
        )

        call = ToolCallMessage(
            id="tc1", name="WriteFile", arguments={"path": "/tmp/file.txt", "content": "data"}
        )

        results = await executor.execute_tool_calls([call], "task-1")

        assert results[0].status == "denied"
        result_data = json.loads(results[0].result_text)
        assert result_data["error"]["code"] == "APPROVAL_DENIED"
        router.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_expired_policy_denies_all(self) -> None:
        """Expired policy → all tools denied."""
        bundle = make_policy_bundle(expired=True)
        enforcer = PolicyEnforcer(bundle)
        router = _make_router()

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            session_id="sess-1",
        )

        for tool_name in ["ReadFile", "WriteFile", "DeleteFile"]:
            call = ToolCallMessage(
                id=f"tc-{tool_name}", name=tool_name, arguments={"path": "/tmp/x"}
            )
            results = await executor.execute_tool_calls([call], "task-1")
            assert results[0].status == "denied", f"{tool_name} should be denied"

    @pytest.mark.asyncio
    async def test_capability_not_granted_denied(self) -> None:
        """Tool whose capability is not in the policy bundle → denied."""
        bundle = make_policy_bundle(capabilities=[{"name": "LLM.Call"}])
        enforcer = PolicyEnforcer(bundle)
        router = _make_router(["ReadFile"])

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            session_id="sess-1",
        )

        call = ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/tmp/test"})
        results = await executor.execute_tool_calls([call], "task-1")
        assert results[0].status == "denied"
        result_data = json.loads(results[0].result_text)
        assert result_data["error"]["code"] == "CAPABILITY_DENIED"
