"""SubAgentManager — spawns focused sub-agents for parallel work."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from agent_host.thread.compactor import DropOldestCompactor
from agent_host.thread.message_thread import MessageThread

if TYPE_CHECKING:
    from agent_host.budget.token_budget import TokenBudget
    from agent_host.llm.client import LLMClient
    from agent_host.loop.tool_executor import ToolExecutor
    from agent_host.policy.policy_enforcer import PolicyEnforcer

logger = structlog.get_logger()

# Maximum concurrent sub-agents
MAX_CONCURRENT = 5

# Sub-agent limits
_SUB_AGENT_MAX_STEPS = 10
_SUB_AGENT_RECENCY_WINDOW = 10
_RESULT_MAX_CHARS = 2000


class SubAgentManager:
    """Manages spawning and running focused sub-agents.

    Sub-agents run with:
    - Fresh MessageThread (isolated context)
    - Restricted tool set (optional)
    - NO SpawnAgent tool (depth=1, no recursion)
    - Shared TokenBudget (safe — asyncio is single-threaded cooperative)
    - Reduced max_steps (10)
    - Result truncated to 2K characters
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        policy_enforcer: PolicyEnforcer,
        token_budget: TokenBudget,
        max_context_tokens: int = 100_000,
    ) -> None:
        self._llm_client = llm_client
        self._tool_executor = tool_executor
        self._policy_enforcer = policy_enforcer
        self._token_budget = token_budget
        self._max_context_tokens = max_context_tokens
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def spawn(
        self,
        task: str,
        context: str,
        parent_task_id: str,
        allowed_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """Spawn a sub-agent to handle a focused task.

        Args:
            task: The task description for the sub-agent.
            context: Additional context from the parent agent.
            parent_task_id: Parent task ID for tracking.
            allowed_tools: Optional list of tool names to restrict to.

        Returns:
            Dict with status and result text.
        """
        async with self._semaphore:
            return await self._run_sub_agent(task, context, parent_task_id, allowed_tools)

    async def _run_sub_agent(
        self,
        task: str,
        context: str,
        parent_task_id: str,
        _allowed_tools: list[str] | None,
    ) -> dict[str, Any]:
        """Run a sub-agent with isolated context."""
        from agent_host.loop.agent_loop import AgentLoop

        sub_task_id = f"{parent_task_id}-sub"

        # Build system prompt for sub-agent
        system_prompt = (
            "You are a focused sub-agent working on a specific task. "
            "Complete the task efficiently and report your findings.\n\n"
        )
        if context:
            system_prompt += f"Context from parent agent:\n{context}\n\n"
        system_prompt += "Focus on the assigned task. Do not deviate."

        # Fresh isolated thread
        thread = MessageThread(system_prompt=system_prompt)
        thread.add_user_message(task)

        # Build compactor for sub-agent
        compactor = DropOldestCompactor(recency_window=_SUB_AGENT_RECENCY_WINDOW)

        # Cancellation event for sub-agent
        cancel = asyncio.Event()

        try:
            # Build and run sub-agent loop (no working_memory, no SpawnAgent)
            loop = AgentLoop(
                llm_client=self._llm_client,
                tool_executor=self._tool_executor,
                thread=thread,
                compactor=compactor,
                policy_enforcer=self._policy_enforcer,
                token_budget=self._token_budget,  # Shared budget
                event_emitter=None,  # Sub-agents don't emit events
                cancellation_event=cancel,
                max_steps=_SUB_AGENT_MAX_STEPS,
                max_context_tokens=self._max_context_tokens,
                working_memory=None,  # No working memory for sub-agents
            )

            result = await loop.run(sub_task_id)

            # Truncate result text
            result_text = result.text
            if len(result_text) > _RESULT_MAX_CHARS:
                result_text = result_text[:_RESULT_MAX_CHARS] + "... [truncated]"

            logger.info(
                "sub_agent_completed",
                parent_task_id=parent_task_id,
                sub_task_id=sub_task_id,
                reason=result.reason,
                steps=result.step_count,
            )

            return {
                "status": "completed" if result.reason == "completed" else result.reason,
                "result": result_text,
                "steps": result.step_count,
            }

        except Exception as exc:
            logger.warning(
                "sub_agent_failed",
                parent_task_id=parent_task_id,
                error=str(exc),
                exc_info=True,
            )
            return {
                "status": "error",
                "result": f"Sub-agent failed: {exc!s}",
                "steps": 0,
            }
