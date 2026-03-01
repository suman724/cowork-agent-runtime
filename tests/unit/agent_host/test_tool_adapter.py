"""Tests for tool_adapter — bridging ToolRouter tools to ADK FunctionTools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from cowork_platform.tool_definition import ToolDefinition
from cowork_platform.tool_result import ToolResult
from google.adk.tools import FunctionTool, LongRunningFunctionTool

from agent_host.agent.tool_adapter import (
    TOOL_CAPABILITY_MAP,
    adapt_tools,
    get_capability_for_tool,
)
from agent_host.policy.policy_enforcer import PolicyEnforcer
from tests.fixtures.policy_bundles import make_policy_bundle, make_restrictive_bundle
from tool_runtime.models import ToolExecutionResult


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
