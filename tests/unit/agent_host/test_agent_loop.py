"""Tests for AgentLoop — full loop with MockLLMClient."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from agent_host.budget.token_budget import TokenBudget
from agent_host.llm.models import LLMResponse
from agent_host.loop.agent_loop import AgentLoop
from agent_host.loop.models import ToolCallResult
from agent_host.thread.compactor import DropOldestCompactor
from agent_host.thread.message_thread import MessageThread
from tests.fixtures.mock_llm import MockLLMClient
from tests.fixtures.policy_bundles import make_policy_bundle


def _make_loop(
    mock_llm: MockLLMClient,
    *,
    max_steps: int = 50,
    max_context_tokens: int = 100_000,
    event_emitter: MagicMock | None = None,
    cancel_event: asyncio.Event | None = None,
    memory_manager: MagicMock | None = None,
    working_memory: MagicMock | None = None,
) -> AgentLoop:
    """Helper to build an AgentLoop with test components."""
    from agent_host.policy.policy_enforcer import PolicyEnforcer

    bundle = make_policy_bundle()
    enforcer = PolicyEnforcer(bundle)
    budget = TokenBudget(max_session_tokens=100_000)
    thread = MessageThread(system_prompt="You are helpful.")
    thread.add_user_message("Test prompt")
    compactor = DropOldestCompactor(recency_window=20)

    # Mock tool executor
    tool_executor = MagicMock()
    tool_executor.get_tool_definitions.return_value = []
    tool_executor.execute_tool_calls = MagicMock(return_value=[])

    return AgentLoop(
        llm_client=mock_llm,  # type: ignore[arg-type]
        tool_executor=tool_executor,
        thread=thread,
        compactor=compactor,
        policy_enforcer=enforcer,
        token_budget=budget,
        event_emitter=event_emitter,
        cancellation_event=cancel_event or asyncio.Event(),
        max_steps=max_steps,
        max_context_tokens=max_context_tokens,
        memory_manager=memory_manager,
        working_memory=working_memory,  # type: ignore[arg-type]
    )


class TestAgentLoopCompletion:
    async def test_simple_text_response(self) -> None:
        """Loop should complete after a single text response with no tool calls."""
        mock = MockLLMClient()
        mock.enqueue_text("Hello!")
        loop = _make_loop(mock)
        result = await loop.run("task-1")
        assert result.reason == "completed"
        assert result.text == "Hello!"
        assert result.step_count == 1

    async def test_tool_call_then_completion(self) -> None:
        """Loop should execute tool call, then complete on text response."""
        mock = MockLLMClient()
        mock.enqueue_tool_call("ReadFile", {"path": "/foo"})
        mock.enqueue_text("Done!")

        loop = _make_loop(mock)
        # Make tool_executor return a result
        tool_result = ToolCallResult(
            tool_call_id="tc-1",
            tool_name="ReadFile",
            status="success",
            result_text='{"status": "success", "output": "file content"}',
        )

        async def mock_execute(calls, task_id, **kwargs):
            return [tool_result]

        loop._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        result = await loop.run("task-1")
        assert result.reason == "completed"
        assert result.step_count == 2

    async def test_multiple_tool_calls(self) -> None:
        """Loop should handle multiple tool call iterations."""
        mock = MockLLMClient()
        mock.enqueue_tool_call("ReadFile", {"path": "/a"}, tool_call_id="tc-1")
        mock.enqueue_tool_call("WriteFile", {"path": "/b", "content": "x"}, tool_call_id="tc-2")
        mock.enqueue_text("All done!")

        loop = _make_loop(mock)

        async def mock_execute(calls, task_id, **kwargs):
            return [
                ToolCallResult(
                    tool_call_id=calls[0].id,
                    tool_name=calls[0].name,
                    status="success",
                    result_text='{"status": "success"}',
                )
            ]

        loop._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        result = await loop.run("task-1")
        assert result.reason == "completed"
        assert result.step_count == 3


class TestAgentLoopStepLimit:
    async def test_max_steps_exceeded(self) -> None:
        """Loop should stop when max_steps is reached."""
        mock = MockLLMClient()
        # Queue up tool calls that never terminate naturally
        for i in range(5):
            mock.enqueue_tool_call("ReadFile", {"path": f"/{i}"}, tool_call_id=f"tc-{i}")

        loop = _make_loop(mock, max_steps=3)

        async def mock_execute(calls, task_id, **kwargs):
            return [
                ToolCallResult(
                    tool_call_id=calls[0].id,
                    tool_name=calls[0].name,
                    status="success",
                    result_text='{"status": "success"}',
                )
            ]

        loop._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        result = await loop.run("task-1")
        assert result.reason == "max_steps_exceeded"
        assert result.step_count == 3

    async def test_step_limit_warning_at_80_percent(self) -> None:
        """Emit step_limit_approaching at 80% of max_steps."""
        mock = MockLLMClient()
        for i in range(5):
            mock.enqueue_tool_call("ReadFile", tool_call_id=f"tc-{i}")
        mock.enqueue_text("Done")

        emitter = MagicMock()
        loop = _make_loop(mock, max_steps=5, event_emitter=emitter)

        async def mock_execute(calls, task_id, **kwargs):
            return [
                ToolCallResult(
                    tool_call_id=calls[0].id,
                    tool_name=calls[0].name,
                    status="success",
                    result_text='{"status": "success"}',
                )
            ]

        loop._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        await loop.run("task-1")
        # 80% of 5 = 4.0, floor = 4
        emitter.emit_step_limit_approaching.assert_called_once_with("task-1", 4, 5)


class TestAgentLoopCancellation:
    async def test_cancellation_before_first_step(self) -> None:
        """Loop should exit immediately if cancel is set before starting."""
        mock = MockLLMClient()
        mock.enqueue_text("should not reach")
        cancel = asyncio.Event()
        cancel.set()

        loop = _make_loop(mock, cancel_event=cancel)
        result = await loop.run("task-1")
        assert result.reason == "cancelled"
        assert mock.call_count == 0

    async def test_cancellation_mid_loop(self) -> None:
        """Loop should check cancel between steps."""
        mock = MockLLMClient()
        cancel = asyncio.Event()

        # First response is a tool call
        mock.enqueue_tool_call("ReadFile", tool_call_id="tc-1")
        # Next would be another tool call, but we'll cancel before
        mock.enqueue_text("should not reach")

        loop = _make_loop(mock, cancel_event=cancel)

        call_count = 0

        async def mock_execute(calls, task_id, **kwargs):
            nonlocal call_count
            call_count += 1
            cancel.set()  # Cancel after first tool execution
            return [
                ToolCallResult(
                    tool_call_id=calls[0].id,
                    tool_name=calls[0].name,
                    status="success",
                    result_text='{"status": "success"}',
                )
            ]

        loop._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        result = await loop.run("task-1")
        assert result.reason == "cancelled"
        assert call_count == 1


class TestAgentLoopBudget:
    async def test_records_token_usage(self) -> None:
        """Loop should record token usage from each LLM response."""
        mock = MockLLMClient()
        mock.enqueue(LLMResponse(text="Hi", stop_reason="stop", input_tokens=100, output_tokens=50))

        loop = _make_loop(mock)
        await loop.run("task-1")
        assert loop._token_budget.input_tokens_used == 100
        assert loop._token_budget.output_tokens_used == 50


class TestAgentLoopStepCallback:
    async def test_on_step_complete_called_each_step(self) -> None:
        """on_step_complete callback should be called after each step."""
        mock = MockLLMClient()
        mock.enqueue_tool_call("ReadFile", {"path": "/a"}, tool_call_id="tc-1")
        mock.enqueue_tool_call("ReadFile", {"path": "/b"}, tool_call_id="tc-2")
        mock.enqueue_text("Done!")

        callback_calls: list[tuple[str, int]] = []

        async def on_step(task_id: str, step: int) -> None:
            callback_calls.append((task_id, step))

        loop = _make_loop(mock)
        loop._on_step_complete = on_step

        async def mock_execute(calls, task_id, **kwargs):
            return [
                ToolCallResult(
                    tool_call_id=calls[0].id,
                    tool_name=calls[0].name,
                    status="success",
                    result_text='{"status": "success"}',
                )
            ]

        loop._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        result = await loop.run("task-1")
        assert result.reason == "completed"
        assert result.step_count == 3
        assert callback_calls == [("task-1", 1), ("task-1", 2), ("task-1", 3)]

    async def test_on_step_complete_failure_does_not_abort_loop(self) -> None:
        """Callback failure should be logged but not abort the agent loop."""
        mock = MockLLMClient()
        mock.enqueue_tool_call("ReadFile", tool_call_id="tc-1")
        mock.enqueue_text("Done!")

        async def failing_callback(task_id: str, step: int) -> None:
            raise RuntimeError("checkpoint write failed")

        loop = _make_loop(mock)
        loop._on_step_complete = failing_callback

        async def mock_execute(calls, task_id, **kwargs):
            return [
                ToolCallResult(
                    tool_call_id=calls[0].id,
                    tool_name=calls[0].name,
                    status="success",
                    result_text='{"status": "success"}',
                )
            ]

        loop._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        result = await loop.run("task-1")
        assert result.reason == "completed"
        assert result.step_count == 2

    async def test_on_step_complete_not_called_when_none(self) -> None:
        """When on_step_complete is None, loop runs normally without callback."""
        mock = MockLLMClient()
        mock.enqueue_text("Hello!")
        loop = _make_loop(mock)
        assert loop._on_step_complete is None
        result = await loop.run("task-1")
        assert result.reason == "completed"


class TestAgentLoopCompaction:
    async def test_compaction_at_90_percent(self) -> None:
        """Loop should compact at 90% of max_context_tokens."""
        mock = MockLLMClient()
        mock.enqueue_text("Done")

        # Use a very small max_context_tokens so the thread is over 90%
        loop = _make_loop(mock, max_context_tokens=10)

        # Stuff the thread with messages to force compaction
        for i in range(10):
            loop._thread.add_user_message(f"Message {i} " + "x" * 100)
            loop._thread.add_assistant_message(f"Response {i} " + "y" * 100)

        emitter = MagicMock()
        loop._event_emitter = emitter

        await loop.run("task-1")
        # With max_context_tokens=10 and 90% budget=9, compaction must have happened
        emitter.emit_context_compacted.assert_called()


class TestAgentLoopMemoryInjection:
    async def test_persistent_memory_injected_into_messages(self) -> None:
        """Memory context should be injected as a system message after the system prompt."""
        mock = MockLLMClient()
        mock.enqueue_text("Done")

        mm = MagicMock()
        mm.render_memory_context.return_value = "# Persistent Memory\n\nUser prefers dark mode"

        loop = _make_loop(mock, memory_manager=mm)
        await loop.run("task-1")

        # Inspect messages sent to LLM
        call_args = mock.last_messages
        assert call_args is not None
        # messages[0] is system prompt, messages[1] should be memory
        memory_msgs = [
            m
            for m in call_args
            if m.get("role") == "system" and "Persistent Memory" in m.get("content", "")
        ]
        assert len(memory_msgs) == 1
        assert "dark mode" in memory_msgs[0]["content"]

    async def test_working_memory_injected_after_persistent_memory(self) -> None:
        """Working memory should be injected after persistent memory."""
        mock = MockLLMClient()
        mock.enqueue_text("Done")

        mm = MagicMock()
        mm.render_memory_context.return_value = "# Persistent Memory\n\nFact 1"

        wm = MagicMock()
        wm.render.return_value = "# Working Memory\n\n- Task 1: pending"
        wm.task_tracker = MagicMock()

        loop = _make_loop(mock, memory_manager=mm, working_memory=wm)
        await loop.run("task-1")

        call_args = mock.last_messages
        assert call_args is not None

        # Find indices of memory and working memory injections
        mem_idx = next(
            i
            for i, m in enumerate(call_args)
            if m.get("role") == "system" and "Persistent Memory" in m.get("content", "")
        )
        wm_idx = next(
            i
            for i, m in enumerate(call_args)
            if m.get("role") == "system" and "Working Memory" in m.get("content", "")
        )
        assert mem_idx < wm_idx

    async def test_no_injection_when_memory_is_empty(self) -> None:
        """No extra system messages when memory returns empty string."""
        mock = MockLLMClient()
        mock.enqueue_text("Done")

        mm = MagicMock()
        mm.render_memory_context.return_value = ""

        loop = _make_loop(mock, memory_manager=mm)
        await loop.run("task-1")

        call_args = mock.last_messages
        assert call_args is not None
        # Only system prompt + user message should be system messages
        system_msgs = [m for m in call_args if m.get("role") == "system"]
        assert len(system_msgs) == 1  # just the system prompt

    async def test_memory_re_read_each_turn(self) -> None:
        """render_memory_context should be called once per loop iteration."""
        mock = MockLLMClient()
        mock.enqueue_tool_call("ReadFile", {"path": "/a"}, tool_call_id="tc-1")
        mock.enqueue_text("Done")

        mm = MagicMock()
        mm.render_memory_context.return_value = "# Persistent Memory\n\nFact"

        loop = _make_loop(mock, memory_manager=mm)

        async def mock_execute(calls, task_id, **kwargs):
            return [
                ToolCallResult(
                    tool_call_id=calls[0].id,
                    tool_name=calls[0].name,
                    status="success",
                    result_text='{"status": "success"}',
                )
            ]

        loop._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        await loop.run("task-1")

        # Should be called once per step (2 steps in this test)
        assert mm.render_memory_context.call_count == 2
