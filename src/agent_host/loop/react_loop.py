"""ReactLoop — linear ReAct agent loop strategy."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import structlog

from agent_host.loop.models import LoopResult
from agent_host.thread.token_counter import estimate_message_tokens

if TYPE_CHECKING:
    from agent_host.llm.models import ToolCallMessage
    from agent_host.loop.loop_runtime import LoopRuntime
    from agent_host.loop.verification import VerificationConfig

logger = structlog.get_logger()


class ReactLoop:
    """Linear ReAct loop: LLM -> tools -> repeat until done.

    Context assembly:
    - System prompt (stable prefix, set at session start)
    - Persistent memory (MEMORY.md) injected after system prompt (semi-stable)
    - Conversation history (grows, prefix is stable for caching)
    - Working memory (task tracker + plan + notes) appended at end (volatile)
    - Error recovery prompts appended when loop detection or reflection triggers fire
    """

    def __init__(
        self,
        harness: LoopRuntime,
        max_steps: int = 50,
        verification: VerificationConfig | None = None,
    ) -> None:
        self._h = harness
        self._max_steps = max_steps
        self._verification = verification

    async def run(self, task_id: str) -> LoopResult:
        """Run the agent loop for a single task."""
        step = 0
        last_text = ""
        verification_injected = False

        while step < self._max_steps:
            step_id = self._h.new_step_id()

            # Check cancellation
            if self._h.is_cancelled():
                logger.info("agent_loop_cancelled", task_id=task_id, step=step)
                return LoopResult(reason="cancelled", step_count=step)

            # 1. Context assembly
            messages = self._build_messages(task_id, step, step_id)
            tools = self._h.get_external_tool_defs() + self._h.get_agent_tool_defs()

            # 2. Emit step_started
            self._h.emit_step_started(task_id, step + 1, step_id)

            # 3. LLM call (policy + budget + streaming)
            def _on_chunk(text: str, _sid: str = step_id) -> None:
                self._h.emit_text_chunk(task_id, text, _sid)

            response = await self._h.call_llm(
                messages,
                tools,
                task_id,
                step_id,
                on_text_chunk=_on_chunk if self._h._event_emitter else None,
            )

            # 4. Record assistant message in thread
            self._h.thread.add_assistant_message(response.text, response.tool_calls)
            last_text = response.text

            # 5. Step counting + events
            step += 1
            self._h.emit_step_completed(task_id, step, step_id)
            await self._h.on_step_complete(task_id, step)

            # 80% warning
            warning_threshold = math.floor(self._max_steps * 0.8)
            if step == warning_threshold:
                self._h.emit_step_limit_approaching(task_id, step, self._max_steps)

            # 6. Natural termination (checked BEFORE hard limit so verification
            #    can extend the step budget)
            if not response.tool_calls and response.stop_reason == "stop":
                # Verification phase: inject verification prompt on first completion
                if self._verification and self._verification.enabled and not verification_injected:
                    verification_injected = True
                    self._max_steps += self._verification.max_verify_steps
                    prompt = self._verification.build_prompt()
                    self._h.thread.add_system_injection(prompt)
                    self._h.emit_verification_started(task_id)
                    continue  # Re-enter loop so LLM can verify

                # Agent confirmed completion (post-verification or no verification)
                if verification_injected:
                    self._h.emit_verification_completed(task_id, passed=True)
                return LoopResult(
                    reason="completed",
                    text=last_text,
                    step_count=step,
                )

            # Hard limit
            if step >= self._max_steps:
                self._h.emit_task_failed(task_id, f"Step limit reached ({step}/{self._max_steps})")
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

            # 7. Execute tool calls
            await self._execute_tools(response.tool_calls, task_id, step_id)

            # 8. Error recovery tracking
            self._track_tool_errors(response.tool_calls)

        return LoopResult(reason="max_steps_exceeded", text=last_text, step_count=step)

    def _build_messages(self, task_id: str, step: int, step_id: str) -> list[dict[str, object]]:
        """Assemble LLM context optimized for prompt caching.

        Ordering (stable prefix first, volatile last):
        1. System prompt (stable — never changes mid-task)
        2. Persistent memory (semi-stable — changes only on SaveMemory)
        3. Conversation history (grows but prefix is stable)
        4. Working memory (volatile — changes every turn, at the END)
        5. Error recovery prompts (conditional — at the very end)
        """
        injection_overhead = 0
        persistent_memory_text: str | None = None
        working_memory_text: str | None = None

        # Persistent memory (MEMORY.md) — semi-stable
        if self._h.memory_manager:
            mem = self._h.memory_manager.render_memory_context()
            if mem:
                persistent_memory_text = mem
                injection_overhead += estimate_message_tokens({"role": "system", "content": mem})

        # Working memory (task tracker + plan + notes) — volatile
        if self._h.working_memory:
            wm = self._h.working_memory.render()
            if wm:
                working_memory_text = wm
                injection_overhead += estimate_message_tokens({"role": "system", "content": wm})

        # Compact conversation history
        compaction_budget = int(self._h.max_context_tokens * 0.9) - injection_overhead
        pre_count = self._h.thread.message_count + 1  # +1 for system prompt
        messages = self._h.thread.build_llm_payload(compaction_budget, self._h.compactor)
        post_count = len(messages)

        # Insert persistent memory right after system prompt (stable prefix)
        if persistent_memory_text and len(messages) > 0:
            messages.insert(1, {"role": "system", "content": persistent_memory_text})

        # Detect compaction
        compacted = any(
            "earlier messages omitted" in m.get("content", "")
            for m in messages
            if m.get("role") == "system" and m is not messages[0]
        )
        if compacted:
            dropped = max(0, pre_count - post_count + 1)
            self._h.emit_context_compacted(task_id, dropped, pre_count, post_count, step_id)

        # Append working memory at the end (volatile — doesn't break cache prefix)
        if working_memory_text:
            messages.append({"role": "system", "content": working_memory_text})

        # Error recovery prompt injection (at the very end)
        er = self._h.error_recovery
        if er.detect_loop():
            prompt = er.build_loop_break_prompt()
            self._h.thread.add_system_injection(prompt)
            messages.append({"role": "system", "content": prompt})
            logger.warning("loop_detected", task_id=task_id, step=step)
        elif er.should_inject_reflection():
            prompt = er.build_reflection_prompt()
            self._h.thread.add_system_injection(prompt)
            messages.append({"role": "system", "content": prompt})
            logger.info("reflection_injected", task_id=task_id, step=step)

        return messages

    async def _execute_tools(
        self, tool_calls: list[ToolCallMessage], task_id: str, step_id: str
    ) -> None:
        """Route tool calls to agent-internal or external execution."""
        agent_calls = []
        external_calls = []
        for tc in tool_calls:
            if self._h.is_agent_tool(tc.name):
                agent_calls.append(tc)
            else:
                external_calls.append(tc)

        # Agent-internal tools
        for ac in agent_calls:
            result_dict = await self._h.execute_agent_tool(ac, task_id)
            result_text = json.dumps(result_dict, default=str)
            self._h.thread.add_tool_result(ac.id, ac.name, result_text)

        # External tools (policy-checked, approval-gated)
        if external_calls:
            results = await self._h.execute_external_tools(external_calls, task_id, step_id)
            for r in results:
                self._h.thread.add_tool_result(
                    r.tool_call_id, r.tool_name, r.result_text, image_url=r.image_url
                )

    def _track_tool_errors(self, tool_calls: list[ToolCallMessage]) -> None:
        """Feed tool results into error recovery tracker."""
        er = self._h.error_recovery
        for tc in tool_calls:
            for msg in reversed(self._h.thread.messages):
                if msg.get("role") == "tool" and msg.get("tool_call_id") == tc.id:
                    content = msg.get("content", "")
                    try:
                        result_data = json.loads(content)
                        status = result_data.get("status", "")
                    except (json.JSONDecodeError, AttributeError):
                        status = "success"
                    if status in ("failed", "denied"):
                        error_msg = str(result_data.get("error", {}).get("message", ""))
                        er.record_tool_failure(tc.name, tc.arguments, error_msg)
                    else:
                        er.record_tool_success(tc.name)
                    break
