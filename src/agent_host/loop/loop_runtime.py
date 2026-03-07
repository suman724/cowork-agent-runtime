"""LoopRuntime — infrastructure primitives for loop strategies."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any

import structlog

from agent_host.loop.error_recovery import ErrorRecovery

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent_host.budget.token_budget import TokenBudget
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.llm.client import LLMClient
    from agent_host.llm.models import LLMResponse, ToolCallMessage
    from agent_host.loop.agent_tools import AgentToolHandler
    from agent_host.loop.models import ToolCallResult
    from agent_host.loop.strategy import LoopStrategy
    from agent_host.loop.tool_executor import ToolExecutor
    from agent_host.memory.memory_manager import MemoryManager
    from agent_host.memory.working_memory import WorkingMemory
    from agent_host.policy.policy_enforcer import PolicyEnforcer
    from agent_host.skills.models import SkillDefinition
    from agent_host.thread.compactor import ContextCompactor
    from agent_host.thread.message_thread import MessageThread

logger = structlog.get_logger()

# Sub-agent defaults
_SUB_AGENT_MAX_STEPS = 10
_SUB_AGENT_RECENCY_WINDOW = 10
_RESULT_MAX_CHARS = 2000

# Skill defaults
_SKILL_RECENCY_WINDOW = 10
_SKILL_RESULT_MAX_CHARS = 4000


class LoopRuntime:
    """Infrastructure primitives for loop strategies.

    Owns all backend service coupling, event emission, and bookkeeping.
    Has no opinions about orchestration order or context assembly.

    Lifetime: one per task. Built by SessionManager, passed to LoopStrategy.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        thread: MessageThread,
        compactor: ContextCompactor,
        policy_enforcer: PolicyEnforcer,
        token_budget: TokenBudget,
        event_emitter: EventEmitter | None,
        cancellation_event: asyncio.Event,
        max_context_tokens: int,
        working_memory: WorkingMemory | None = None,
        memory_manager: MemoryManager | None = None,
        error_recovery: ErrorRecovery | None = None,
        agent_tool_handler: AgentToolHandler | None = None,
        on_step_complete: Callable[[str, int], Awaitable[None]] | None = None,
        default_sub_agent_factory: Callable[[LoopRuntime], LoopStrategy] | None = None,
        skills: list[SkillDefinition] | None = None,
        max_concurrent_sub_agents: int = 5,
    ) -> None:
        self._llm_client = llm_client
        self._tool_executor = tool_executor
        self._thread = thread
        self._compactor = compactor
        self._policy_enforcer = policy_enforcer
        self._token_budget = token_budget
        self._event_emitter = event_emitter
        self._cancel = cancellation_event
        self._max_context_tokens = max_context_tokens
        self._working_memory = working_memory
        self._memory_manager = memory_manager
        self._error_recovery = error_recovery or ErrorRecovery()
        self._agent_tool_handler = agent_tool_handler
        self._on_step_complete = on_step_complete
        self._default_sub_agent_factory = default_sub_agent_factory
        self._skills = {s.name: s for s in (skills or [])}
        self._sub_agent_semaphore = asyncio.Semaphore(max_concurrent_sub_agents)

        # Wire sub-agent/skill callbacks into the agent tool handler
        if self._agent_tool_handler:
            self._agent_tool_handler._spawn_sub_agent = self.spawn_sub_agent
            self._agent_tool_handler._execute_skill = self.execute_skill

    # ── Primitives ──────────────────────────────────────────────

    def is_cancelled(self) -> bool:
        """Check if the task has been cancelled."""
        return self._cancel.is_set()

    def new_step_id(self) -> str:
        """Generate a new UUID v4 step ID."""
        return str(uuid.uuid4())

    # ── LLM ─────────────────────────────────────────────────────

    async def call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        task_id: str,
        _step_id: str,
        on_text_chunk: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Policy check + budget pre-check + stream LLM + record usage.

        Raises: CapabilityDeniedError, LLMBudgetExceededError, LLMGatewayError

        ``_step_id`` is accepted for future per-step tracing but currently unused.
        """
        self._policy_enforcer.check_llm_call()
        self._token_budget.pre_check()

        response = await self._llm_client.stream_chat(
            messages,
            tools,
            task_id=task_id,
            on_text_chunk=on_text_chunk,
        )

        self._token_budget.record_usage(response.input_tokens, response.output_tokens)
        return response

    # ── Tools ───────────────────────────────────────────────────

    def get_external_tool_defs(self) -> list[dict[str, Any]]:
        """Tool definitions from ToolRouter (policy-filtered)."""
        return self._tool_executor.get_tool_definitions()

    def get_agent_tool_defs(self) -> list[dict[str, Any]]:
        """Agent-internal tool definitions (TaskTracker, CreatePlan, memory, etc.)."""
        if self._agent_tool_handler:
            return self._agent_tool_handler.get_tool_definitions()
        return []

    async def execute_external_tools(
        self,
        tool_calls: list[ToolCallMessage],
        task_id: str,
        step_id: str,
    ) -> list[ToolCallResult]:
        """Execute tools through ToolExecutor (policy, approval, events)."""
        return await self._tool_executor.execute_tool_calls(tool_calls, task_id, step_id=step_id)

    async def execute_agent_tool(
        self,
        tool_call: ToolCallMessage,
        task_id: str,
    ) -> dict[str, Any]:
        """Execute an agent-internal tool (no policy, no ToolRouter).

        Also emits tool_requested / tool_completed events.
        """
        if not self._agent_tool_handler:
            return {"status": "error", "message": "Agent tools not available"}

        # Classify tool type for event emission
        if tool_call.name == "SpawnAgent":
            agent_tool_type = "sub_agent"
        elif tool_call.name.startswith("Skill_"):
            agent_tool_type = "skill"
        else:
            agent_tool_type = "agent"

        # Emit tool_requested
        if self._event_emitter:
            self._event_emitter.emit_tool_requested(
                tool_name=tool_call.name,
                capability="",
                arguments=tool_call.arguments,
                tool_call_id=tool_call.id,
                tool_type=agent_tool_type,
            )

        result_dict = await self._agent_tool_handler.execute(
            tool_call.name, tool_call.arguments, task_id=task_id
        )

        # Emit tool_completed
        if self._event_emitter:
            status = (
                result_dict.get("status", "success") if isinstance(result_dict, dict) else "success"
            )
            result_text = json.dumps(result_dict, default=str)
            self._event_emitter.emit_tool_completed(
                tool_name=tool_call.name,
                status=str(status),
                tool_call_id=tool_call.id,
                result=result_text,
                tool_type=agent_tool_type,
            )

        return result_dict

    def is_agent_tool(self, tool_name: str) -> bool:
        """Check if a tool name is agent-internal."""
        if self._agent_tool_handler:
            return self._agent_tool_handler.is_agent_tool(tool_name)
        return False

    # ── Events ──────────────────────────────────────────────────

    def emit_step_started(self, task_id: str, step: int, step_id: str) -> None:
        """Emit step_started event."""
        if self._event_emitter:
            self._event_emitter.emit_step_started(task_id, step, step_id=step_id)

    def emit_step_completed(self, task_id: str, step: int, step_id: str) -> None:
        """Emit step_completed event."""
        if self._event_emitter:
            self._event_emitter.emit_step_completed(task_id, step, step_id=step_id)

    def emit_text_chunk(self, task_id: str, text: str, step_id: str) -> None:
        """Emit text_chunk event."""
        if self._event_emitter:
            self._event_emitter.emit_text_chunk(task_id, text, step_id=step_id)

    def emit_step_limit_approaching(self, task_id: str, step: int, max_steps: int) -> None:
        """Emit step_limit_approaching event."""
        if self._event_emitter:
            self._event_emitter.emit_step_limit_approaching(task_id, step, max_steps)

    def emit_task_failed(self, task_id: str, reason: str) -> None:
        """Emit task_failed event."""
        if self._event_emitter:
            self._event_emitter.emit_task_failed(task_id, reason=reason)

    def emit_context_compacted(
        self, task_id: str, dropped: int, pre_count: int, post_count: int, step_id: str
    ) -> None:
        """Emit context_compacted event."""
        if self._event_emitter:
            self._event_emitter.emit_context_compacted(
                task_id, dropped, pre_count, post_count, step_id=step_id
            )

    # ── Checkpoint ──────────────────────────────────────────────

    async def on_step_complete(self, task_id: str, step: int) -> None:
        """Invoke the checkpoint callback. Errors are logged, never raised."""
        if self._on_step_complete:
            try:
                await self._on_step_complete(task_id, step)
            except Exception:
                logger.warning(
                    "on_step_complete_callback_failed",
                    task_id=task_id,
                    step=step,
                    exc_info=True,
                )

    # ── Sub-agents ──────────────────────────────────────────────

    async def spawn_sub_agent(
        self,
        task: str,
        context: str,
        parent_task_id: str,
        max_steps: int = _SUB_AGENT_MAX_STEPS,
        strategy_factory: Callable[[LoopRuntime], LoopStrategy] | None = None,
    ) -> dict[str, Any]:
        """Spawn a sub-agent with an isolated thread but shared token budget.

        Builds a child LoopRuntime with:
          - Shared: LLMClient, TokenBudget, PolicyEnforcer (same instances)
          - Fresh: MessageThread, ContextCompactor, ErrorRecovery
          - Excluded: WorkingMemory, MemoryManager, sub-agent spawning (no recursion)
        Enforces concurrency via semaphore.
        """
        async with self._sub_agent_semaphore:
            return await self._run_sub_agent(
                task, context, parent_task_id, max_steps, strategy_factory
            )

    async def _run_sub_agent(
        self,
        task: str,
        context: str,
        parent_task_id: str,
        max_steps: int,
        strategy_factory: Callable[[LoopRuntime], LoopStrategy] | None,
    ) -> dict[str, Any]:
        """Run a sub-agent with isolated context."""
        from agent_host.thread.compactor import DropOldestCompactor
        from agent_host.thread.message_thread import MessageThread

        sub_task_id = f"{parent_task_id}-sub"

        # Build system prompt
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
        compactor = DropOldestCompactor(recency_window=_SUB_AGENT_RECENCY_WINDOW)
        cancel = asyncio.Event()

        try:
            child_harness = LoopRuntime(
                llm_client=self._llm_client,
                tool_executor=self._tool_executor,
                thread=thread,
                compactor=compactor,
                policy_enforcer=self._policy_enforcer,
                token_budget=self._token_budget,
                event_emitter=None,  # Sub-agents don't emit events
                cancellation_event=cancel,
                max_context_tokens=self._max_context_tokens,
                # Excluded: no working_memory, no memory_manager, no agent_tool_handler
                # No sub-agent spawning (depth=1)
            )

            # Use provided factory or default
            factory = strategy_factory or self._default_sub_agent_factory
            if factory:
                strategy = factory(child_harness)
            else:
                # Late import to avoid circular dependency
                from agent_host.loop.react_loop import ReactLoop

                strategy = ReactLoop(child_harness, max_steps=max_steps)

            result = await strategy.run(sub_task_id)

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

    # ── Skills ──────────────────────────────────────────────────

    async def execute_skill(
        self,
        skill: SkillDefinition,
        arguments: dict[str, Any],
        parent_task_id: str,
        strategy_factory: Callable[[LoopRuntime], LoopStrategy] | None = None,
    ) -> dict[str, Any]:
        """Execute a skill as a focused sub-conversation."""
        from agent_host.skills.skill_loader import SkillLoader, substitute_arguments
        from agent_host.thread.compactor import DropOldestCompactor
        from agent_host.thread.message_thread import MessageThread

        task_id = f"{parent_task_id}-skill-{skill.name}"

        # Load full content on demand
        skill = SkillLoader.load_skill_content(skill)

        # Apply argument substitution
        prompt_content = substitute_arguments(skill.prompt_content, arguments)

        # Build system prompt
        system_prompt = (
            f"You are executing the '{skill.name}' skill.\n\nSkill: {skill.description}\n\n"
        )
        if prompt_content:
            system_prompt += f"{prompt_content}\n\n"
        system_prompt += "Complete the task and report your results clearly."

        # Build user message
        parts = [f"Execute the '{skill.name}' skill with these inputs:"]
        for key, value in arguments.items():
            parts.append(f"- {key}: {value}")
        user_message = "\n".join(parts)

        # Fresh thread
        thread = MessageThread(system_prompt=system_prompt)
        thread.add_user_message(user_message)
        compactor = DropOldestCompactor(recency_window=_SKILL_RECENCY_WINDOW)
        cancel = asyncio.Event()

        try:
            child_harness = LoopRuntime(
                llm_client=self._llm_client,
                tool_executor=self._tool_executor,
                thread=thread,
                compactor=compactor,
                policy_enforcer=self._policy_enforcer,
                token_budget=self._token_budget,
                event_emitter=None,
                cancellation_event=cancel,
                max_context_tokens=self._max_context_tokens,
            )

            factory = strategy_factory or self._default_sub_agent_factory
            if factory:
                strategy = factory(child_harness)
            else:
                from agent_host.loop.react_loop import ReactLoop

                strategy = ReactLoop(child_harness, max_steps=skill.max_steps)

            result = await strategy.run(task_id)

            result_text = result.text
            if len(result_text) > _SKILL_RESULT_MAX_CHARS:
                result_text = result_text[:_SKILL_RESULT_MAX_CHARS] + "... [truncated]"

            logger.info(
                "skill_completed",
                skill_name=skill.name,
                parent_task_id=parent_task_id,
                reason=result.reason,
                steps=result.step_count,
            )

            return {
                "status": "completed" if result.reason == "completed" else result.reason,
                "result": result_text,
                "skill": skill.name,
                "steps": result.step_count,
            }

        except Exception as exc:
            logger.warning(
                "skill_failed",
                skill_name=skill.name,
                parent_task_id=parent_task_id,
                error=str(exc),
                exc_info=True,
            )
            return {
                "status": "error",
                "result": f"Skill '{skill.name}' failed: {exc!s}",
                "skill": skill.name,
                "steps": 0,
            }

    # ── Read-only Properties ────────────────────────────────────

    @property
    def thread(self) -> MessageThread:
        """The conversation thread."""
        return self._thread

    @property
    def compactor(self) -> ContextCompactor:
        """Context compaction strategy."""
        return self._compactor

    @property
    def working_memory(self) -> WorkingMemory | None:
        """Working memory (task tracker, plan, notes)."""
        return self._working_memory

    @property
    def memory_manager(self) -> MemoryManager | None:
        """Persistent memory manager."""
        return self._memory_manager

    @property
    def error_recovery(self) -> ErrorRecovery:
        """Error recovery tracker."""
        return self._error_recovery

    @property
    def token_budget(self) -> TokenBudget:
        """Token budget."""
        return self._token_budget

    @property
    def max_context_tokens(self) -> int:
        """Maximum context window size."""
        return self._max_context_tokens

    @property
    def policy_enforcer(self) -> PolicyEnforcer:
        """Policy enforcer."""
        return self._policy_enforcer
