"""Tests for skill execution via LoopRuntime.execute_skill()."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from agent_host.budget.token_budget import TokenBudget
from agent_host.llm.models import LLMResponse
from agent_host.loop.loop_runtime import LoopRuntime
from agent_host.loop.models import ToolCallResult
from agent_host.policy.policy_enforcer import PolicyEnforcer
from agent_host.skills.models import SkillDefinition
from tests.fixtures.mock_llm import MockLLMClient
from tests.fixtures.policy_bundles import make_policy_bundle


def _make_skill(
    name: str = "test_skill",
    description: str = "A test skill.",
    max_steps: int = 15,
    tool_subset: list[str] | None = None,
    prompt_content: str = "",
) -> SkillDefinition:
    """Create a SkillDefinition for testing."""
    return SkillDefinition(
        name=name,
        description=description,
        prompt_content=prompt_content,
        tool_subset=tool_subset,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        max_steps=max_steps,
    )


def _make_loop_runtime(
    mock_llm: MockLLMClient | None = None,
    max_context_tokens: int = 100_000,
) -> LoopRuntime:
    """Build a LoopRuntime for testing skill execution."""
    if mock_llm is None:
        mock_llm = MockLLMClient()
        mock_llm.enqueue_text("Skill completed successfully.")

    from agent_host.thread.compactor import DropOldestCompactor
    from agent_host.thread.message_thread import MessageThread

    bundle = make_policy_bundle()
    enforcer = PolicyEnforcer(bundle)
    budget = TokenBudget(max_session_tokens=100_000)

    tool_executor = MagicMock()
    tool_executor.get_tool_definitions.return_value = []
    tool_executor.execute_tool_calls = AsyncMock(return_value=[])

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
    )


class TestSkillExecution:
    async def test_execute_simple_skill(self) -> None:
        """Skill should complete a simple task."""
        mock = MockLLMClient()
        mock.enqueue_text("Found 3 matches in the codebase.")
        runtime = _make_loop_runtime(mock)
        skill = _make_skill()

        result = await runtime.execute_skill(
            skill=skill,
            arguments={"query": "find auth endpoints"},
            parent_task_id="task-1",
        )

        assert result["status"] == "completed"
        assert "3 matches" in result["result"]
        assert result["skill"] == "test_skill"
        assert result["steps"] == 1

    async def test_execute_with_tool_calls(self) -> None:
        """Skill should handle tool calls within its loop."""
        mock = MockLLMClient()
        mock.enqueue_tool_call("ReadFile", {"path": "/src/main.py"})
        mock.enqueue_text("File contains the auth handler.")

        runtime = _make_loop_runtime(mock)

        tool_result = ToolCallResult(
            tool_call_id="tc-1",
            tool_name="ReadFile",
            status="success",
            result_text='{"status": "success", "output": "def auth(): pass"}',
        )

        async def mock_execute(calls, task_id, **kwargs):
            return [tool_result]

        runtime._tool_executor.execute_tool_calls = mock_execute  # type: ignore[assignment]

        skill = _make_skill()
        result = await runtime.execute_skill(
            skill=skill,
            arguments={"query": "read auth file"},
            parent_task_id="task-1",
        )

        assert result["status"] == "completed"
        assert result["steps"] == 2

    async def test_execute_with_prompt_content(self) -> None:
        """Skill with prompt_content should include it in system prompt."""
        mock = MockLLMClient()
        mock.enqueue_text("Search complete.")
        runtime = _make_loop_runtime(mock)
        skill = _make_skill(
            prompt_content="Focus on Python files only.",
        )

        await runtime.execute_skill(
            skill=skill,
            arguments={"query": "test"},
            parent_task_id="task-1",
        )

        # Verify system prompt in the LLM call
        call = mock.call_log[0]
        system_msg = call["messages"][0]
        assert system_msg["role"] == "system"
        assert "Focus on Python files only." in system_msg["content"]

    async def test_argument_substitution_in_prompt(self) -> None:
        """$ARGUMENTS placeholders should be substituted in prompt_content."""
        mock = MockLLMClient()
        mock.enqueue_text("Done.")
        runtime = _make_loop_runtime(mock)
        skill = _make_skill(
            prompt_content="Search for $ARGUMENTS[0] in the codebase.",
        )

        await runtime.execute_skill(
            skill=skill,
            arguments={"query": "authentication"},
            parent_task_id="task-1",
        )

        call = mock.call_log[0]
        system_msg = call["messages"][0]
        assert "Search for authentication in the codebase." in system_msg["content"]
        assert "$ARGUMENTS" not in system_msg["content"]

    async def test_result_truncation(self) -> None:
        """Results longer than 4K chars should be truncated."""
        mock = MockLLMClient()
        mock.enqueue_text("x" * 8000)
        runtime = _make_loop_runtime(mock)
        skill = _make_skill()

        result = await runtime.execute_skill(
            skill=skill,
            arguments={"query": "generate long output"},
            parent_task_id="task-1",
        )

        assert len(result["result"]) <= 4100  # 4000 + "... [truncated]" overhead
        assert result["result"].endswith("... [truncated]")

    async def test_error_handling(self) -> None:
        """Skill should handle errors gracefully."""
        mock = MockLLMClient()
        mock.enqueue(RuntimeError("LLM gateway timeout"))
        runtime = _make_loop_runtime(mock)
        skill = _make_skill()

        result = await runtime.execute_skill(
            skill=skill,
            arguments={"query": "fail"},
            parent_task_id="task-1",
        )

        assert result["status"] == "error"
        assert "LLM gateway timeout" in result["result"]
        assert result["skill"] == "test_skill"
        assert result["steps"] == 0

    async def test_max_steps_enforcement(self) -> None:
        """Skill should respect its max_steps limit."""
        mock = MockLLMClient()
        for i in range(20):
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

        skill = _make_skill(max_steps=5)
        result = await runtime.execute_skill(
            skill=skill,
            arguments={"query": "many steps"},
            parent_task_id="task-1",
        )

        assert result["status"] == "max_steps_exceeded"
        assert result["steps"] == 5

    async def test_task_id_format(self) -> None:
        """Task ID should include parent task ID and skill name."""
        mock = MockLLMClient()
        mock.enqueue_text("Done.")
        runtime = _make_loop_runtime(mock)
        skill = _make_skill(name="search_codebase")

        await runtime.execute_skill(
            skill=skill,
            arguments={"query": "test"},
            parent_task_id="parent-123",
        )

        call = mock.call_log[0]
        assert call["task_id"] == "parent-123-skill-search_codebase"


class TestSkillExecutionTokenBudget:
    async def test_shared_token_budget(self) -> None:
        """Skill should use the shared token budget."""
        mock = MockLLMClient()
        mock.enqueue(
            LLMResponse(text="Done", stop_reason="stop", input_tokens=100, output_tokens=50)
        )

        runtime = _make_loop_runtime(mock)
        budget = runtime.token_budget

        skill = _make_skill()
        await runtime.execute_skill(
            skill=skill,
            arguments={"query": "test"},
            parent_task_id="task-1",
        )

        assert budget.input_tokens_used == 100
        assert budget.output_tokens_used == 50
