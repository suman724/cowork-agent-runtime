"""Tests for verification phase in the agent loop."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from agent_host.budget.token_budget import TokenBudget
from agent_host.loop.loop_runtime import LoopRuntime
from agent_host.loop.models import ToolCallResult
from agent_host.loop.react_loop import ReactLoop
from agent_host.loop.verification import VerificationConfig
from agent_host.thread.compactor import DropOldestCompactor
from agent_host.thread.message_thread import MessageThread
from tests.fixtures.mock_llm import MockLLMClient
from tests.fixtures.policy_bundles import make_policy_bundle


def _make_loop(
    mock_llm: MockLLMClient,
    *,
    max_steps: int = 50,
    verification: VerificationConfig | None = None,
    event_emitter: MagicMock | None = None,
) -> ReactLoop:
    from agent_host.policy.policy_enforcer import PolicyEnforcer

    bundle = make_policy_bundle()
    enforcer = PolicyEnforcer(bundle)
    budget = TokenBudget(max_session_tokens=100_000)
    thread = MessageThread(system_prompt="You are helpful.")
    thread.add_user_message("Test prompt")
    compactor = DropOldestCompactor(recency_window=20)

    tool_executor = MagicMock()
    tool_executor.get_tool_definitions.return_value = []
    tool_executor.execute_tool_calls = MagicMock(return_value=[])

    harness = LoopRuntime(
        llm_client=mock_llm,  # type: ignore[arg-type]
        tool_executor=tool_executor,
        thread=thread,
        compactor=compactor,
        policy_enforcer=enforcer,
        token_budget=budget,
        event_emitter=event_emitter,
        cancellation_event=asyncio.Event(),
        max_context_tokens=100_000,
    )

    return ReactLoop(harness, max_steps=max_steps, verification=verification)


class TestVerificationConfig:
    def test_default_prompt(self) -> None:
        """Default prompt should contain verification instructions."""
        config = VerificationConfig()
        prompt = config.build_prompt()
        assert "VERIFICATION" in prompt
        assert "Re-read the original request" in prompt

    def test_custom_instructions_prompt(self) -> None:
        """Custom instructions should be embedded in the prompt."""
        config = VerificationConfig(custom_instructions="Check CSV headers")
        prompt = config.build_prompt()
        assert "Check CSV headers" in prompt
        assert "Re-read the original request" in prompt

    def test_disabled_config(self) -> None:
        """Disabled config should still build prompt if called."""
        config = VerificationConfig(enabled=False)
        assert config.build_prompt()  # still returns a prompt


class TestVerificationPhase:
    async def test_verification_injected_on_completion(self) -> None:
        """Verification prompt should be injected when agent first completes."""
        mock = MockLLMClient()
        # First: agent says "Done" (triggers verification)
        # Second: agent confirms "Verified" (passes verification)
        mock.enqueue_text("Done!")
        mock.enqueue_text("Verified, all looks good.")

        verification = VerificationConfig(enabled=True, max_verify_steps=3)
        loop = _make_loop(mock, verification=verification)
        result = await loop.run("task-1")

        assert result.reason == "completed"
        assert result.step_count == 2
        # Verify the thread has a system injection with VERIFICATION
        messages = loop._h.thread.messages
        verification_msgs = [
            m
            for m in messages
            if m.get("role") == "system" and "VERIFICATION" in m.get("content", "")
        ]
        assert len(verification_msgs) == 1

    async def test_verification_with_fix(self) -> None:
        """Agent should be able to use tools to fix issues during verification."""
        mock = MockLLMClient()
        # First: agent says "Done" (triggers verification)
        # Second: agent uses a tool to fix something
        # Third: agent confirms "Fixed and verified"
        mock.enqueue_text("Done!")
        mock.enqueue_tool_call("ReadFile", {"path": "/output.csv"}, tool_call_id="tc-v1")
        mock.enqueue_text("All verified and correct.")

        verification = VerificationConfig(enabled=True, max_verify_steps=5)
        loop = _make_loop(mock, verification=verification)

        async def mock_execute(calls, task_id, **kwargs):
            return [
                ToolCallResult(
                    tool_call_id=calls[0].id,
                    tool_name=calls[0].name,
                    status="success",
                    result_text='{"status": "success", "output": "data"}',
                )
            ]

        loop._h._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        result = await loop.run("task-1")
        assert result.reason == "completed"
        assert result.step_count == 3

    async def test_no_verification_when_disabled(self) -> None:
        """No verification prompt when verification is disabled."""
        mock = MockLLMClient()
        mock.enqueue_text("Done!")

        verification = VerificationConfig(enabled=False)
        loop = _make_loop(mock, verification=verification)
        result = await loop.run("task-1")

        assert result.reason == "completed"
        assert result.step_count == 1

    async def test_no_verification_when_none(self) -> None:
        """No verification when config is None."""
        mock = MockLLMClient()
        mock.enqueue_text("Done!")

        loop = _make_loop(mock, verification=None)
        result = await loop.run("task-1")

        assert result.reason == "completed"
        assert result.step_count == 1

    async def test_verification_extends_step_budget(self) -> None:
        """max_steps should increase by max_verify_steps when verification kicks in."""
        mock = MockLLMClient()
        # Fill up to max_steps-1 with tool calls, then complete
        mock.enqueue_tool_call("ReadFile", tool_call_id="tc-1")
        mock.enqueue_text("Done!")  # step 2 — triggers verification
        mock.enqueue_text("Verified!")  # step 3 — completes

        verification = VerificationConfig(enabled=True, max_verify_steps=3)
        loop = _make_loop(mock, max_steps=2, verification=verification)

        async def mock_execute(calls, task_id, **kwargs):
            return [
                ToolCallResult(
                    tool_call_id=calls[0].id,
                    tool_name=calls[0].name,
                    status="success",
                    result_text='{"status": "success"}',
                )
            ]

        loop._h._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        result = await loop.run("task-1")
        # Without verification step budget extension, this would be max_steps_exceeded
        assert result.reason == "completed"
        assert result.step_count == 3

    async def test_verification_events_emitted(self) -> None:
        """Should emit verification_started and verification_completed events."""
        mock = MockLLMClient()
        mock.enqueue_text("Done!")
        mock.enqueue_text("Verified!")

        emitter = MagicMock()
        verification = VerificationConfig(enabled=True)
        loop = _make_loop(mock, verification=verification, event_emitter=emitter)
        await loop.run("task-1")

        emitter.emit_verification_started.assert_called_once_with("task-1")
        emitter.emit_verification_completed.assert_called_once_with("task-1", passed=True)
