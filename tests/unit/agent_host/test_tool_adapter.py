"""Tests for tool_adapter — bridging ToolRouter tools to ADK FunctionTools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from cowork_platform.tool_definition import ToolDefinition
from cowork_platform.tool_result import ToolResult
from google.adk.tools import FunctionTool, LongRunningFunctionTool

from agent_host.agent.artifact_store import PendingArtifactStore
from agent_host.agent.file_change_tracker import FileChangeTracker
from agent_host.agent.tool_adapter import (
    TOOL_CAPABILITY_MAP,
    ToolExecutionDeps,
    adapt_tools,
    get_capability_for_tool,
)
from agent_host.approval.approval_gate import ApprovalGate
from agent_host.policy.policy_enforcer import PolicyEnforcer
from tests.fixtures.policy_bundles import make_policy_bundle, make_restrictive_bundle
from tool_runtime.models import ArtifactData, ToolExecutionResult


class TestGetCapabilityForTool:
    def test_known_tools(self) -> None:
        assert get_capability_for_tool("ReadFile") == "File.Read"
        assert get_capability_for_tool("WriteFile") == "File.Write"
        assert get_capability_for_tool("DeleteFile") == "File.Delete"
        assert get_capability_for_tool("RunCommand") == "Shell.Exec"
        assert get_capability_for_tool("HttpRequest") == "Network.Http"

    def test_unknown_tool(self) -> None:
        assert get_capability_for_tool("UnknownTool") == ""

    def test_all_capabilities_mapped(self) -> None:
        assert len(TOOL_CAPABILITY_MAP) == 5


class TestAdaptTools:
    def test_creates_function_tools(self) -> None:
        """Standard tools become FunctionTool."""
        router = MagicMock()
        router.get_available_tools.return_value = [
            ToolDefinition(
                toolName="ReadFile",
                description="Read a file",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ]
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)

        tools = adapt_tools(router, enforcer)
        assert len(tools) == 1
        assert isinstance(tools[0], FunctionTool)

    def test_creates_long_running_for_approval(self) -> None:
        """Tools requiring approval become LongRunningFunctionTool."""
        router = MagicMock()
        router.get_available_tools.return_value = [
            ToolDefinition(
                toolName="ReadFile",
                description="Read a file",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ]
        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)

        tools = adapt_tools(router, enforcer)
        assert len(tools) == 1
        assert isinstance(tools[0], LongRunningFunctionTool)

    def test_adapts_all_available_tools(self) -> None:
        """All tools from router are adapted."""
        router = MagicMock()
        router.get_available_tools.return_value = [
            ToolDefinition(
                toolName="ReadFile",
                description="Read",
                inputSchema={"type": "object"},
            ),
            ToolDefinition(
                toolName="WriteFile",
                description="Write",
                inputSchema={"type": "object"},
            ),
            ToolDefinition(
                toolName="RunCommand",
                description="Run",
                inputSchema={"type": "object"},
            ),
        ]
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)

        tools = adapt_tools(router, enforcer)
        assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_tool_function_calls_router(self) -> None:
        """Adapted tool function calls tool_router.execute()."""
        mock_result = ToolExecutionResult(
            tool_result=ToolResult(
                toolName="ReadFile",
                sessionId="sess-1",
                taskId="task-1",
                stepId="step-1",
                status="succeeded",
                outputText="file contents",
            ),
        )
        router = MagicMock()
        router.get_available_tools.return_value = [
            ToolDefinition(
                toolName="ReadFile",
                description="Read a file",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ]
        router.execute = AsyncMock(return_value=mock_result)

        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)

        tools = adapt_tools(router, enforcer, session_id="sess-1", task_id="task-1")
        assert len(tools) == 1

        # Get the underlying function and call it
        fn = tools[0].func
        result = await fn(path="/tmp/test.txt")
        assert result["status"] == "succeeded"
        assert result["output"] == "file contents"
        router.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_function_handles_execution_error(self) -> None:
        """Adapted tool function catches exceptions and returns error dict."""
        router = MagicMock()
        router.get_available_tools.return_value = [
            ToolDefinition(
                toolName="ReadFile",
                description="Read a file",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ]
        router.execute = AsyncMock(side_effect=RuntimeError("disk error"))

        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)

        tools = adapt_tools(router, enforcer, session_id="sess-1", task_id="task-1")
        fn = tools[0].func
        result = await fn(path="/tmp/test.txt")

        assert result["status"] == "failed"
        assert "disk error" in result["error"]["message"]
        assert result["error"]["code"] == "TOOL_EXECUTION_FAILED"


class TestToolExecutionDeps:
    @pytest.mark.asyncio
    async def test_approval_gate_blocks_denied(self) -> None:
        """When ApprovalGate denies, tool returns APPROVAL_DENIED error."""
        router = MagicMock()
        router.get_available_tools.return_value = [
            ToolDefinition(
                toolName="WriteFile",
                description="Write a file",
                inputSchema={"type": "object"},
            ),
        ]
        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        gate = ApprovalGate()
        emitter = MagicMock()

        deps = ToolExecutionDeps(
            tool_router=router,
            policy_enforcer=enforcer,
            event_emitter=emitter,
            approval_gate=gate,
            approval_timeout=0.05,  # fast timeout for test
            session_id="sess-1",
            task_id="task-1",
        )

        tools = adapt_tools(router, enforcer, deps=deps)
        fn = tools[0].func

        # No one delivers approval → timeout → denied
        result = await fn(path="/tmp/test.txt", content="hello")
        assert result["status"] == "denied"
        assert result["error"]["code"] == "APPROVAL_DENIED"
        router.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_approval_gate_approves(self) -> None:
        """When ApprovalGate approves, tool executes normally."""
        import asyncio

        mock_result = ToolExecutionResult(
            tool_result=ToolResult(
                toolName="WriteFile",
                sessionId="sess-1",
                taskId="task-1",
                stepId="step-1",
                status="succeeded",
                outputText="file written",
            ),
        )
        router = MagicMock()
        router.get_available_tools.return_value = [
            ToolDefinition(
                toolName="WriteFile",
                description="Write a file",
                inputSchema={"type": "object"},
            ),
        ]
        router.execute = AsyncMock(return_value=mock_result)
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

        async def _approve() -> None:
            await asyncio.sleep(0.01)
            # Get approval_id from emitter call
            call_kwargs = emitter.emit_approval_requested.call_args[1]
            gate.deliver(call_kwargs["approval_id"], "approved")

        _task = asyncio.create_task(_approve())  # noqa: RUF006
        result = await fn(path="/tmp/test.txt", content="hello")
        assert result["status"] == "succeeded"
        router.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_artifact_store_populated(self) -> None:
        """Artifacts from ToolExecutionResult are stored in PendingArtifactStore."""
        artifact = ArtifactData(
            artifact_type="tool_output",
            artifact_name="big-output.txt",
            data=b"lots of data",
            media_type="text/plain",
        )
        mock_result = ToolExecutionResult(
            tool_result=ToolResult(
                toolName="ReadFile",
                sessionId="sess-1",
                taskId="task-1",
                stepId="step-1",
                status="succeeded",
                outputText="truncated",
            ),
            artifacts=[artifact],
        )
        router = MagicMock()
        router.get_available_tools.return_value = [
            ToolDefinition(
                toolName="ReadFile",
                description="Read a file",
                inputSchema={"type": "object"},
            ),
        ]
        router.execute = AsyncMock(return_value=mock_result)
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        store = PendingArtifactStore()

        deps = ToolExecutionDeps(
            tool_router=router,
            policy_enforcer=enforcer,
            artifact_store=store,
            session_id="sess-1",
            task_id="task-1",
        )

        tools = adapt_tools(router, enforcer, deps=deps)
        fn = tools[0].func
        await fn(path="/tmp/test.txt")

        popped = store.pop("ReadFile")
        assert len(popped) == 1
        assert popped[0].artifact_name == "big-output.txt"

    @pytest.mark.asyncio
    async def test_file_change_tracker_records_write(self, tmp_path: object) -> None:
        """WriteFile tool records file changes in FileChangeTracker."""
        mock_result = ToolExecutionResult(
            tool_result=ToolResult(
                toolName="WriteFile",
                sessionId="sess-1",
                taskId="task-1",
                stepId="step-1",
                status="succeeded",
                outputText="written",
            ),
        )
        router = MagicMock()
        router.get_available_tools.return_value = [
            ToolDefinition(
                toolName="WriteFile",
                description="Write a file",
                inputSchema={"type": "object"},
            ),
        ]

        # Write the file as a side effect of execute
        file_path = str(Path(str(tmp_path)) / "test.txt")

        async def _mock_execute(request, context):  # type: ignore[no-untyped-def]
            Path(file_path).write_text("new content")
            return mock_result

        router.execute = AsyncMock(side_effect=_mock_execute)
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        tracker = FileChangeTracker()

        deps = ToolExecutionDeps(
            tool_router=router,
            policy_enforcer=enforcer,
            file_change_tracker=tracker,
            session_id="sess-1",
            task_id="task-1",
        )

        tools = adapt_tools(router, enforcer, deps=deps)
        fn = tools[0].func
        await fn(path=file_path, content="new content")

        preview = tracker.get_patch_preview("task-1")
        assert len(preview["files"]) == 1
        assert preview["files"][0]["status"] == "added"  # new file
