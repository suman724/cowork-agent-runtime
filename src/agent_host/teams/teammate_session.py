"""TeammateSessionManager — lightweight agent loop for team members."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from agent_host.budget.token_budget import TokenBudget
from agent_host.loop.agent_tools import AgentToolHandler
from agent_host.loop.loop_runtime import LoopRuntime
from agent_host.loop.react_loop import ReactLoop
from agent_host.loop.tool_executor import ToolExecutor
from agent_host.memory.working_memory import WorkingMemory
from agent_host.thread.compactor import DropOldestCompactor
from agent_host.thread.message_thread import MessageThread

if TYPE_CHECKING:
    from agent_host.coordination.protocols import (
        ContextInjectionStrategy,
        ToolProviderStrategy,
    )
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.llm.client import LLMClient
    from agent_host.loop.models import LoopResult
    from agent_host.policy.policy_enforcer import PolicyEnforcer
    from tool_runtime import ToolRouter

logger = structlog.get_logger()

_TEAMMATE_SYSTEM_PROMPT = """You are a teammate in a multi-agent team working on a shared project.

Team: {team_name}
Your name: {teammate_name}
Your role: {role}

## Working Directory
You are working in: {workspace_dir}
You share this workspace with other teammates. Stick to the files relevant to your \
assigned tasks. Check the task list to see what others are working on to avoid conflicts.

## Coordination
- Use TeamTaskList to see what needs to be done
- Use TeamTaskUpdate to pick up tasks and report completion
- Use SendTeamMessage to communicate with teammates or the lead
- Save your work frequently

## Guidelines
- Focus on your assigned tasks
- When blocked, message the relevant teammate or the lead
- When your current task is done, check the task list for more work
- If the task list is empty and you have no messages, let the lead know you are idle
"""


class TeammateSessionManager:
    """Lightweight session manager for teammate agents.

    Shares: LLMClient, PolicyEnforcer, PolicyBundle, ToolRouter, workspace dir.
    Fresh: MessageThread, TokenBudget, WorkingMemory.
    """

    def __init__(
        self,
        name: str,
        role: str,
        team_name: str,
        llm_client: LLMClient,
        policy_enforcer: PolicyEnforcer,
        tool_router: ToolRouter,
        workspace_dir: str | None,
        tool_provider: ToolProviderStrategy,
        context_injector: ContextInjectionStrategy,
        budget: int = 100_000,
        max_steps: int = 50,
        max_context_tokens: int = 100_000,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self._team_name = team_name
        self._llm_client = llm_client
        self._policy_enforcer = policy_enforcer
        self._tool_router = tool_router
        self._workspace_dir = workspace_dir
        self._tool_provider = tool_provider
        self._context_injector = context_injector
        self._max_steps = max_steps
        self._max_context_tokens = max_context_tokens
        self._event_emitter = event_emitter

        # Fresh per-teammate resources
        self._token_budget = TokenBudget(max_session_tokens=budget)
        self._working_memory = WorkingMemory()
        self._cancel_event = asyncio.Event()

        # Build system prompt
        system_prompt = _TEAMMATE_SYSTEM_PROMPT.format(
            team_name=team_name,
            teammate_name=name,
            role=role,
            workspace_dir=workspace_dir or "(no workspace)",
        )
        self._thread = MessageThread(system_prompt=system_prompt)

        # Task handle for cancellation
        self._task: asyncio.Task[LoopResult | None] | None = None

    @property
    def system_prompt(self) -> str:
        """Return the teammate's system prompt (for testing)."""
        return self._thread._system_prompt

    async def run(self, initial_prompt: str) -> LoopResult | None:
        """Run the teammate agent loop."""
        task_id = f"teammate_{self.name}"

        # Add initial prompt as user message
        self._thread.add_user_message(initial_prompt)

        try:
            from tool_runtime.models import ExecutionContext

            exec_context: ExecutionContext | None = None
            if self._workspace_dir:
                exec_context = ExecutionContext(working_directory=self._workspace_dir)

            tool_executor = ToolExecutor(
                tool_router=self._tool_router,
                policy_enforcer=self._policy_enforcer,
                execution_context=exec_context,
            )

            agent_tool_handler = AgentToolHandler(
                self._working_memory,
                workspace_dir=self._workspace_dir,
                tool_provider=self._tool_provider,
                agent_role="teammate",
            )

            compactor = DropOldestCompactor(recency_window=10)

            loop_runtime = LoopRuntime(
                llm_client=self._llm_client,
                tool_executor=tool_executor,
                thread=self._thread,
                compactor=compactor,
                policy_enforcer=self._policy_enforcer,
                token_budget=self._token_budget,
                event_emitter=self._event_emitter,
                cancellation_event=self._cancel_event,
                max_context_tokens=self._max_context_tokens,
                working_memory=self._working_memory,
                agent_tool_handler=agent_tool_handler,
                workspace_dir=self._workspace_dir,
                context_injector=self._context_injector,
            )

            strategy = ReactLoop(loop_runtime, max_steps=self._max_steps)
            result = await strategy.run(task_id)

            logger.info(
                "teammate_loop_completed",
                name=self.name,
                reason=result.reason,
                steps=result.step_count,
            )
            return result

        except asyncio.CancelledError:
            logger.info("teammate_cancelled", name=self.name)
            return None
        except Exception:
            logger.exception("teammate_loop_error", name=self.name)
            return None

    def cancel(self) -> None:
        """Signal the teammate to stop."""
        self._cancel_event.set()
        if self._task is not None:
            self._task.cancel()
