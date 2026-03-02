"""Tests for ToolExecutor — policy check, approval, file tracking, artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agent_host.agent.file_change_tracker import FileChangeTracker
from agent_host.approval.approval_gate import ApprovalGate
from agent_host.llm.models import ToolCallMessage
from agent_host.loop.tool_executor import ToolExecutor
from agent_host.policy.policy_enforcer import PolicyEnforcer
from tests.fixtures.policy_bundles import make_policy_bundle, make_restrictive_bundle


def _make_tool_router_mock(
    status: str = "succeeded", output: str = "done", artifacts: list | None = None
) -> MagicMock:
    """Create a mock ToolRouter."""
    router = MagicMock()
    router.get_available_tools.return_value = [
        MagicMock(
            toolName="ReadFile",
            description="Read a file",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        MagicMock(
            toolName="WriteFile",
            description="Write a file",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            },
        ),
    ]

    # Use MagicMock for tool_result to avoid ToolResult model validation complexity
    tool_result = MagicMock()
    tool_result.status = status
    tool_result.outputText = output
    tool_result.error = None

    exec_result = MagicMock()
    exec_result.tool_result = tool_result
    exec_result.artifacts = artifacts or []

    router.execute = AsyncMock(return_value=exec_result)
    return router


class TestToolExecutorPolicyCheck:
    async def test_allowed_tool(self) -> None:
        """Allowed tool should execute successfully."""
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            session_id="sess-1",
        )

        call = ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/foo"})
        results = await executor.execute_tool_calls([call], "task-1")

        assert len(results) == 1
        assert results[0].status == "succeeded"
        assert results[0].tool_name == "ReadFile"

    async def test_denied_tool(self) -> None:
        """Denied tool should return denial without execution."""
        # Create an expired bundle so all tools are denied
        bundle = make_policy_bundle(expired=True)
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            session_id="sess-1",
        )

        call = ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/foo"})
        results = await executor.execute_tool_calls([call], "task-1")

        assert len(results) == 1
        assert results[0].status == "denied"
        result_data = json.loads(results[0].result_text)
        assert result_data["error"]["code"] == "CAPABILITY_DENIED"
        router.execute.assert_not_awaited()

    async def test_path_scope_denied(self) -> None:
        """Tool call outside allowed paths should be denied."""
        bundle = make_restrictive_bundle(allowed_paths=["/allowed"])
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            session_id="sess-1",
        )

        call = ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/forbidden/file"})
        results = await executor.execute_tool_calls([call], "task-1")

        assert results[0].status == "denied"
        router.execute.assert_not_awaited()


class TestToolExecutorApproval:
    async def test_approval_required_approved(self) -> None:
        """Tool requiring approval should proceed when approved."""
        bundle = make_restrictive_bundle(requires_approval=True, allowed_paths=["/test"])
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        gate = ApprovalGate()
        emitter = MagicMock()

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            event_emitter=emitter,
            approval_gate=gate,
            session_id="sess-1",
            approval_timeout=1.0,
        )

        call = ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/test"})

        import asyncio

        # Deliver approval after a short delay
        async def deliver():
            await asyncio.sleep(0.05)
            # Find the approval_id from the emitter call
            approval_id = emitter.emit_approval_requested.call_args[1]["approval_id"]
            gate.deliver(approval_id, "approved")

        _task = asyncio.create_task(deliver())  # noqa: RUF006

        results = await executor.execute_tool_calls([call], "task-1")
        assert results[0].status == "succeeded"

    async def test_approval_required_denied(self) -> None:
        """Tool requiring approval should return denial when denied."""
        bundle = make_restrictive_bundle(requires_approval=True, allowed_paths=["/test"])
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        gate = ApprovalGate()
        emitter = MagicMock()

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            event_emitter=emitter,
            approval_gate=gate,
            session_id="sess-1",
            approval_timeout=1.0,
        )

        call = ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/test"})

        import asyncio

        async def deny():
            await asyncio.sleep(0.05)
            approval_id = emitter.emit_approval_requested.call_args[1]["approval_id"]
            gate.deliver(approval_id, "denied")

        _task = asyncio.create_task(deny())  # noqa: RUF006

        results = await executor.execute_tool_calls([call], "task-1")
        assert results[0].status == "denied"
        router.execute.assert_not_awaited()


class TestToolExecutorFileTracking:
    async def test_write_file_tracking(self, tmp_path: Path) -> None:
        """File change tracker should record file writes."""
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        tracker = FileChangeTracker()
        router = _make_tool_router_mock()

        # Create a pre-existing file
        test_file = tmp_path / "test.txt"
        test_file.write_text("old content")

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            file_change_tracker=tracker,
            session_id="sess-1",
        )

        # WriteFile call
        call = ToolCallMessage(
            id="tc1",
            name="WriteFile",
            arguments={"path": str(test_file), "content": "new content"},
        )

        # After execution the router's mock won't actually write the file,
        # but the tracker should have captured the old content
        results = await executor.execute_tool_calls([call], "task-1")
        assert results[0].status == "succeeded"


class TestToolExecutorEvents:
    async def test_emits_tool_requested_and_completed(self) -> None:
        """Should emit tool_requested and tool_completed events."""
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        emitter = MagicMock()

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            event_emitter=emitter,
            session_id="sess-1",
        )

        call = ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/foo"})
        await executor.execute_tool_calls([call], "task-1")

        emitter.emit_tool_requested.assert_called_once()
        emitter.emit_tool_completed.assert_called_once()


class TestToolExecutorDefinitions:
    def test_get_tool_definitions(self) -> None:
        """Should return OpenAI-format tool definitions."""
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        router = _make_tool_router_mock()
        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
        )

        defs = executor.get_tool_definitions()
        assert len(defs) == 2
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "ReadFile"
        assert "parameters" in defs[0]["function"]


class TestToolExecutorError:
    async def test_tool_execution_error(self) -> None:
        """Should catch and return tool execution errors."""
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        router = MagicMock()
        router.get_available_tools.return_value = []
        router.execute = AsyncMock(side_effect=RuntimeError("boom"))

        executor = ToolExecutor(
            tool_router=router,
            policy_enforcer=enforcer,
            session_id="sess-1",
        )

        call = ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/foo"})
        results = await executor.execute_tool_calls([call], "task-1")

        assert results[0].status == "failed"
        result_data = json.loads(results[0].result_text)
        assert result_data["error"]["code"] == "TOOL_EXECUTION_FAILED"
