"""AgentLoop — the core LLM → tool execution → repeat loop."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import structlog

from agent_host.loop.agent_tools import AgentToolHandler
from agent_host.loop.error_recovery import ErrorRecovery
from agent_host.loop.models import LoopResult
from agent_host.thread.token_counter import estimate_message_tokens

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable

    from agent_host.budget.token_budget import TokenBudget
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.llm.client import LLMClient
    from agent_host.loop.sub_agent import SubAgentManager
    from agent_host.loop.tool_executor import ToolExecutor
    from agent_host.memory.memory_manager import MemoryManager
    from agent_host.memory.working_memory import WorkingMemory
    from agent_host.policy.policy_enforcer import PolicyEnforcer
    from agent_host.skills.models import SkillDefinition
    from agent_host.skills.skill_executor import SkillExecutor
    from agent_host.thread.compactor import ContextCompactor
    from agent_host.thread.message_thread import MessageThread

logger = structlog.get_logger()


class AgentLoop:
    """Runs the agent loop: stream LLM → execute tools → repeat until done.

    Replaces the ADK ``Runner.run_async()`` black box with a transparent,
    extensible loop that we fully control.
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
        max_steps: int = 50,
        max_context_tokens: int = 100_000,
        working_memory: WorkingMemory | None = None,
        sub_agent_manager: SubAgentManager | None = None,
        skill_executor: SkillExecutor | None = None,
        skills: list[SkillDefinition] | None = None,
        on_step_complete: Callable[[str, int], Awaitable[None]] | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._tool_executor = tool_executor
        self._thread = thread
        self._compactor = compactor
        self._policy_enforcer = policy_enforcer
        self._token_budget = token_budget
        self._event_emitter = event_emitter
        self._cancel = cancellation_event
        self._max_steps = max_steps
        self._max_context_tokens = max_context_tokens
        self._working_memory = working_memory
        self._memory_manager = memory_manager
        self._agent_tool_handler = (
            AgentToolHandler(
                working_memory,
                sub_agent_manager=sub_agent_manager,
                skill_executor=skill_executor,
                skills=skills,
                memory_manager=memory_manager,
            )
            if working_memory
            else None
        )
        self._on_step_complete = on_step_complete
        self._error_recovery = ErrorRecovery()

    async def run(self, task_id: str) -> LoopResult:
        """Run the agent loop for a single task.

        Returns LoopResult indicating how the loop ended.
        """
        step = 0
        last_text = ""

        while step < self._max_steps:
            # Check cancellation
            if self._cancel.is_set():
                logger.info("agent_loop_cancelled", task_id=task_id, step=step)
                return LoopResult(reason="cancelled", step_count=step)

            # 1. Pre-compute memory injection texts and estimate their token overhead
            mem_text = ""
            wm_text = ""
            injection_overhead = 0

            if self._memory_manager:
                mem_text = self._memory_manager.render_memory_context()
                if mem_text:
                    injection_overhead += estimate_message_tokens(
                        {"role": "system", "content": mem_text}
                    )

            if self._working_memory:
                wm_text = self._working_memory.render()
                if wm_text:
                    injection_overhead += estimate_message_tokens(
                        {"role": "system", "content": wm_text}
                    )

            # Build messages (compact at 90% of context limit, minus injection overhead)
            compaction_budget = int(self._max_context_tokens * 0.9) - injection_overhead
            pre_count = self._thread.message_count + 1  # +1 for system prompt
            messages = self._thread.build_llm_payload(compaction_budget, self._compactor)
            post_count = len(messages)

            # 1b. Inject persistent memory + working memory after system prompt.
            inject_offset = 1  # after system prompt (messages[0])

            if mem_text and len(messages) > 0:
                messages.insert(inject_offset, {"role": "system", "content": mem_text})
                inject_offset += 1

            if wm_text and len(messages) > 0:
                messages.insert(inject_offset, {"role": "system", "content": wm_text})

            # 1d. Inject error recovery prompts if needed (Wave 5)
            if self._error_recovery.detect_loop():
                loop_prompt = self._error_recovery.build_loop_break_prompt()
                self._thread.add_system_injection(loop_prompt)
                messages.append({"role": "system", "content": loop_prompt})
                logger.warning("loop_detected", task_id=task_id, step=step)
            elif self._error_recovery.should_inject_reflection():
                reflect_prompt = self._error_recovery.build_reflection_prompt()
                self._thread.add_system_injection(reflect_prompt)
                messages.append({"role": "system", "content": reflect_prompt})
                logger.info("reflection_injected", task_id=task_id, step=step)

            # Detect compaction via marker message presence
            compacted = any(
                "earlier messages omitted" in m.get("content", "")
                for m in messages
                if m.get("role") == "system" and m is not messages[0]
            )
            if compacted and self._event_emitter:
                # Marker replaces N dropped messages (marker adds 1, so net = pre - post + 1)
                dropped = max(0, pre_count - post_count + 1)
                self._event_emitter.emit_context_compacted(task_id, dropped, pre_count, post_count)

            # 2. Emit step_started event
            if self._event_emitter:
                self._event_emitter.emit_step_started(task_id, step + 1)

            # 3. Check policy + budget, then stream LLM call
            self._policy_enforcer.check_llm_call()
            self._token_budget.pre_check()

            # Build combined tool definitions (external + agent-internal)
            tool_defs = self._tool_executor.get_tool_definitions()
            if self._agent_tool_handler:
                tool_defs = tool_defs + self._agent_tool_handler.get_tool_definitions()

            response = await self._llm_client.stream_chat(
                messages,
                tool_defs,
                task_id=task_id,
                on_text_chunk=(
                    (lambda t: self._event_emitter.emit_text_chunk(task_id, t))  # type: ignore[union-attr]
                    if self._event_emitter
                    else None
                ),
            )

            # Record token usage
            self._token_budget.record_usage(response.input_tokens, response.output_tokens)

            # 4. Record assistant message in thread
            self._thread.add_assistant_message(response.text, response.tool_calls)
            last_text = response.text

            # 5. Step counting + limit enforcement
            step += 1

            # Emit step_completed
            if self._event_emitter:
                self._event_emitter.emit_step_completed(task_id, step)

            # Per-step checkpoint callback
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

            # 80% warning
            warning_threshold = math.floor(self._max_steps * 0.8)
            if step == warning_threshold and self._event_emitter:
                self._event_emitter.emit_step_limit_approaching(task_id, step, self._max_steps)

            # Hard limit
            if step >= self._max_steps:
                if self._event_emitter:
                    self._event_emitter.emit_task_failed(
                        task_id,
                        reason=f"Step limit reached ({step}/{self._max_steps})",
                    )
                logger.warning(
                    "step_limit_reached",
                    task_id=task_id,
                    step_count=step,
                    max_steps=self._max_steps,
                )
                return LoopResult(
                    reason="max_steps_exceeded",
                    text=last_text,
                    step_count=step,
                )

            # 6. Natural termination (no tool calls, stop reason = "stop")
            if not response.tool_calls and response.stop_reason == "stop":
                return LoopResult(
                    reason="completed",
                    text=last_text,
                    step_count=step,
                )

            # 7. Execute tool calls — route agent tools internally, rest through ToolExecutor
            agent_calls = []
            external_calls = []
            for tc in response.tool_calls:
                if self._agent_tool_handler and self._agent_tool_handler.is_agent_tool(tc.name):
                    agent_calls.append(tc)
                else:
                    external_calls.append(tc)

            # Execute agent-internal tools (no policy, no ToolRouter)
            for ac in agent_calls:
                handler = self._agent_tool_handler
                if handler is None:
                    err_msg = "agent_tool_handler is None but agent_calls is non-empty"
                    raise RuntimeError(err_msg)

                # Classify tool type for event emission
                if ac.name == "SpawnAgent":
                    agent_tool_type = "sub_agent"
                elif ac.name.startswith("Skill_"):
                    agent_tool_type = "skill"
                else:
                    agent_tool_type = "agent"

                # Emit tool_requested for agent-internal tools
                if self._event_emitter:
                    self._event_emitter.emit_tool_requested(
                        tool_name=ac.name,
                        capability="",
                        arguments=ac.arguments,
                        tool_call_id=ac.id,
                        tool_type=agent_tool_type,
                    )

                result_dict = await handler.execute(ac.name, ac.arguments, task_id=task_id)
                result_text = json.dumps(result_dict, default=str)
                self._thread.add_tool_result(ac.id, ac.name, result_text)

                # Emit tool_completed for agent-internal tools
                if self._event_emitter:
                    status = (
                        result_dict.get("status", "success")
                        if isinstance(result_dict, dict)
                        else "success"
                    )
                    self._event_emitter.emit_tool_completed(
                        tool_name=ac.name,
                        status=str(status),
                        tool_call_id=ac.id,
                        result=result_text,
                        tool_type=agent_tool_type,
                    )

            # Execute external tools (policy-checked, approval-gated)
            if external_calls:
                results = await self._tool_executor.execute_tool_calls(external_calls, task_id)
                for r in results:
                    self._thread.add_tool_result(
                        r.tool_call_id, r.tool_name, r.result_text, image_url=r.image_url
                    )

            # 8. Error recovery tracking (Wave 5)
            for tc in response.tool_calls:
                # Find the corresponding result in the thread
                for msg in reversed(self._thread.messages):
                    if msg.get("role") == "tool" and msg.get("tool_call_id") == tc.id:
                        content = msg.get("content", "")
                        try:
                            result_data = json.loads(content)
                            status = result_data.get("status", "")
                        except (json.JSONDecodeError, AttributeError):
                            status = "success"  # Non-JSON content is considered success
                        if status in ("failed", "denied"):
                            error_msg = str(result_data.get("error", {}).get("message", ""))
                            self._error_recovery.record_tool_failure(
                                tc.name, tc.arguments, error_msg
                            )
                        else:
                            self._error_recovery.record_tool_success(tc.name)
                        break

        return LoopResult(reason="max_steps_exceeded", text=last_text, step_count=step)
