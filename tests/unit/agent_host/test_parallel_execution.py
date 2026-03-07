"""Tests for parallel tool execution grouping in ToolExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from agent_host.llm.models import ToolCallMessage
from agent_host.loop.tool_executor import ToolExecutor
from tests.fixtures.policy_bundles import make_policy_bundle


def _make_tool_router_mock() -> MagicMock:
    """Create a mock ToolRouter that returns success for all tools."""
    router = MagicMock()
    router.get_available_tools.return_value = []

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


def _make_executor(router: MagicMock | None = None) -> ToolExecutor:
    from agent_host.policy.policy_enforcer import PolicyEnforcer

    bundle = make_policy_bundle()
    enforcer = PolicyEnforcer(bundle)
    return ToolExecutor(
        tool_router=router or _make_tool_router_mock(),
        policy_enforcer=enforcer,
        session_id="sess-1",
    )


class TestPartitionParallelGroups:
    def test_all_reads_single_group(self) -> None:
        """All ReadFile calls should be grouped into one parallel batch."""
        executor = _make_executor()
        calls = [
            ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/a"}),
            ToolCallMessage(id="tc2", name="ReadFile", arguments={"path": "/b"}),
            ToolCallMessage(id="tc3", name="ReadFile", arguments={"path": "/c"}),
        ]
        groups = executor._partition_parallel_groups(calls)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_serial_tool_breaks_batch(self) -> None:
        """RunCommand should break a parallel batch."""
        executor = _make_executor()
        calls = [
            ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/a"}),
            ToolCallMessage(id="tc2", name="RunCommand", arguments={"command": "ls"}),
            ToolCallMessage(id="tc3", name="ReadFile", arguments={"path": "/b"}),
        ]
        groups = executor._partition_parallel_groups(calls)
        assert len(groups) == 3
        assert len(groups[0]) == 1  # ReadFile /a
        assert groups[1][0].name == "RunCommand"
        assert groups[2][0].name == "ReadFile"

    def test_writes_to_different_paths_parallel(self) -> None:
        """WriteFile calls to different paths can be parallelized."""
        executor = _make_executor()
        calls = [
            ToolCallMessage(id="tc1", name="WriteFile", arguments={"path": "/a"}),
            ToolCallMessage(id="tc2", name="WriteFile", arguments={"path": "/b"}),
        ]
        groups = executor._partition_parallel_groups(calls)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_writes_to_same_path_serial(self) -> None:
        """WriteFile calls to the same path should serialize."""
        executor = _make_executor()
        calls = [
            ToolCallMessage(id="tc1", name="WriteFile", arguments={"path": "/a"}),
            ToolCallMessage(id="tc2", name="WriteFile", arguments={"path": "/a"}),
        ]
        groups = executor._partition_parallel_groups(calls)
        assert len(groups) == 2

    def test_mixed_reads_and_fetches(self) -> None:
        """ReadFile and FetchUrl should batch together."""
        executor = _make_executor()
        calls = [
            ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/a"}),
            ToolCallMessage(id="tc2", name="FetchUrl", arguments={"url": "http://x"}),
            ToolCallMessage(id="tc3", name="GrepFiles", arguments={"pattern": "foo"}),
        ]
        groups = executor._partition_parallel_groups(calls)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_delete_always_serial(self) -> None:
        """DeleteFile should always be serialized."""
        executor = _make_executor()
        calls = [
            ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/a"}),
            ToolCallMessage(id="tc2", name="DeleteFile", arguments={"path": "/b"}),
        ]
        groups = executor._partition_parallel_groups(calls)
        assert len(groups) == 2

    def test_single_tool_call(self) -> None:
        """Single tool call should be one group."""
        executor = _make_executor()
        calls = [ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/a"})]
        groups = executor._partition_parallel_groups(calls)
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_empty_calls(self) -> None:
        """Empty calls should return empty groups."""
        executor = _make_executor()
        groups = executor._partition_parallel_groups([])
        assert groups == []


class TestParallelExecution:
    async def test_parallel_reads_preserve_order(self) -> None:
        """Results from parallel execution should preserve original call order."""
        router = _make_tool_router_mock()
        executor = _make_executor(router)

        calls = [
            ToolCallMessage(id="tc1", name="ReadFile", arguments={"path": "/a"}),
            ToolCallMessage(id="tc2", name="ReadFile", arguments={"path": "/b"}),
            ToolCallMessage(id="tc3", name="ReadFile", arguments={"path": "/c"}),
        ]

        results = await executor.execute_tool_calls(calls, "task-1")
        assert len(results) == 3
        # All should succeed
        assert all(r.status == "succeeded" for r in results)
        # Router should have been called 3 times
        assert router.execute.await_count == 3

    async def test_serial_tools_execute_sequentially(self) -> None:
        """Non-parallelizable tools should execute one at a time."""
        call_order: list[str] = []
        router = _make_tool_router_mock()

        original_execute = router.execute

        async def tracking_execute(request, context):
            call_order.append(request.toolName)
            return await original_execute(request, context)

        router.execute = tracking_execute
        executor = _make_executor(router)

        calls = [
            ToolCallMessage(id="tc1", name="RunCommand", arguments={"command": "a"}),
            ToolCallMessage(id="tc2", name="RunCommand", arguments={"command": "b"}),
        ]

        results = await executor.execute_tool_calls(calls, "task-1")
        assert len(results) == 2
        assert call_order == ["RunCommand", "RunCommand"]
