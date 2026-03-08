"""Tests for sub-agent spawning via LoopRuntime.spawn_sub_agent()."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from agent_host.budget.token_budget import TokenBudget
from agent_host.llm.models import LLMResponse
from agent_host.loop.loop_runtime import LoopRuntime
from agent_host.loop.models import ToolCallResult
from agent_host.policy.policy_enforcer import PolicyEnforcer
from tests.fixtures.mock_llm import MockLLMClient
from tests.fixtures.policy_bundles import make_policy_bundle

# Re-export constant for concurrency test
MAX_CONCURRENT = 5


def _make_loop_runtime(
    mock_llm: MockLLMClient | None = None,
    max_context_tokens: int = 100_000,
    max_concurrent_sub_agents: int = MAX_CONCURRENT,
) -> LoopRuntime:
    """Build a LoopRuntime for testing sub-agent spawning."""
    if mock_llm is None:
        mock_llm = MockLLMClient()
        mock_llm.enqueue_text("Sub-agent done!")

    from agent_host.thread.compactor import DropOldestCompactor
    from agent_host.thread.message_thread import MessageThread

    bundle = make_policy_bundle()
    enforcer = PolicyEnforcer(bundle)
    budget = TokenBudget(max_session_tokens=100_000)

    tool_executor = MagicMock()
    tool_executor.get_tool_definitions.return_value = []
    tool_executor.execute_tool_calls = AsyncMock(return_value=[])
    tool_executor.plan_mode_locked = False

    thread = MessageThread(system_prompt="test")
    compactor = DropOldestCompactor(recency_window=10)

    return LoopRuntime(
        llm_client=mock_llm,  # type: ignore[arg-type]
        tool_executor=tool_executor,
        thread=thread,
        compactor=compactor,
        policy_enforcer=enforcer,
        token_budget=budget,
        event_emitter=None,
        cancellation_event=asyncio.Event(),
        max_context_tokens=max_context_tokens,
        max_concurrent_sub_agents=max_concurrent_sub_agents,
    )


class TestSubAgentSpawn:
    async def test_spawn_simple_task(self) -> None:
        """Sub-agent should complete a simple text task."""
        mock = MockLLMClient()
        mock.enqueue_text("The answer is 42.")
        runtime = _make_loop_runtime(mock)

        result = await runtime.spawn_sub_agent(
            task="What is the answer?",
            context="We're computing the answer to everything.",
            parent_task_id="task-1",
        )

        assert result["status"] == "completed"
        assert "42" in result["result"]
        assert result["steps"] == 1

    async def test_spawn_with_tool_calls(self) -> None:
        """Sub-agent should handle tool calls within its loop."""
        mock = MockLLMClient()
        mock.enqueue_tool_call("ReadFile", {"path": "/test"})
        mock.enqueue_text("File content found!")

        runtime = _make_loop_runtime(mock)

        tool_result = ToolCallResult(
            tool_call_id="tc-1",
            tool_name="ReadFile",
            status="success",
            result_text='{"status": "success", "output": "file data"}',
        )

        async def mock_execute(calls, task_id, **kwargs):
            return [tool_result]

        runtime._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        result = await runtime.spawn_sub_agent(
            task="Read the test file",
            context="",
            parent_task_id="task-1",
        )

        assert result["status"] == "completed"
        assert result["steps"] == 2

    async def test_result_truncation(self) -> None:
        """Results longer than 2K chars should be truncated."""
        mock = MockLLMClient()
        mock.enqueue_text("x" * 5000)
        runtime = _make_loop_runtime(mock)

        result = await runtime.spawn_sub_agent(
            task="Generate long text", context="", parent_task_id="task-1"
        )
        assert len(result["result"]) <= 2100  # 2000 + "[truncated]" overhead
        assert result["result"].endswith("... [truncated]")

    async def test_error_handling(self) -> None:
        """Sub-agent should handle errors gracefully."""
        mock = MockLLMClient()
        mock.enqueue(RuntimeError("LLM exploded"))
        runtime = _make_loop_runtime(mock)

        result = await runtime.spawn_sub_agent(
            task="Do something", context="", parent_task_id="task-1"
        )
        assert result["status"] == "error"
        assert "LLM exploded" in result["result"]

    async def test_max_steps_enforcement(self) -> None:
        """Sub-agent should respect its max_steps limit."""
        mock = MockLLMClient()
        for i in range(15):
            mock.enqueue_tool_call("ReadFile", {"path": f"/{i}"}, tool_call_id=f"tc-{i}")

        runtime = _make_loop_runtime(mock)

        async def mock_execute(calls, task_id, **kwargs):
            return [
                ToolCallResult(
                    tool_call_id=calls[0].id,
                    tool_name=calls[0].name,
                    status="success",
                    result_text='{"status": "success"}',
                )
            ]

        runtime._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        result = await runtime.spawn_sub_agent(
            task="Do many things", context="", parent_task_id="task-1"
        )
        assert result["status"] == "max_steps_exceeded"
        assert result["steps"] == 10  # _SUB_AGENT_MAX_STEPS


class TestSubAgentConcurrency:
    async def test_concurrency_limit(self) -> None:
        """Should enforce MAX_CONCURRENT sub-agents via semaphore."""
        active_count = 0
        max_active = 0

        original_run = LoopRuntime._run_sub_agent

        async def tracking_run(self, task, context, parent_task_id, max_steps, strategy_factory):
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.01)
            active_count -= 1
            return {"status": "completed", "result": "done", "steps": 1}

        LoopRuntime._run_sub_agent = tracking_run  # type: ignore[assignment]
        try:
            mock = MockLLMClient()
            runtime = _make_loop_runtime(mock)

            tasks = [
                runtime.spawn_sub_agent(task=f"Task {i}", context="", parent_task_id="t-1")
                for i in range(MAX_CONCURRENT + 3)
            ]
            await asyncio.gather(*tasks)

            assert max_active <= MAX_CONCURRENT
        finally:
            LoopRuntime._run_sub_agent = original_run  # type: ignore[assignment]


class TestSubAgentTokenBudget:
    async def test_shared_token_budget(self) -> None:
        """Sub-agent should share parent's token budget."""
        mock = MockLLMClient()
        mock.enqueue(
            LLMResponse(text="Done", stop_reason="stop", input_tokens=50, output_tokens=25)
        )

        runtime = _make_loop_runtime(mock)
        budget = runtime.token_budget

        await runtime.spawn_sub_agent(task="Quick task", context="", parent_task_id="task-1")
        assert budget.input_tokens_used == 50
        assert budget.output_tokens_used == 25
