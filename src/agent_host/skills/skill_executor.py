"""SkillExecutor — executes a skill as a focused sub-conversation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from agent_host.skills.skill_loader import SkillLoader, substitute_arguments
from agent_host.thread.compactor import DropOldestCompactor
from agent_host.thread.message_thread import MessageThread

if TYPE_CHECKING:
    from agent_host.budget.token_budget import TokenBudget
    from agent_host.llm.client import LLMClient
    from agent_host.loop.tool_executor import ToolExecutor
    from agent_host.policy.policy_enforcer import PolicyEnforcer
    from agent_host.skills.models import SkillDefinition

logger = structlog.get_logger()

_SKILL_RECENCY_WINDOW = 10
_RESULT_MAX_CHARS = 4000


class SkillExecutor:
    """Executes skills as focused sub-conversations.

    Each skill runs with:
    - Custom system prompt (base + skill prompt_content with $ARGUMENTS substitution)
    - Restricted tool set (if specified by skill)
    - Dedicated MessageThread
    - Skill-specific max_steps
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

    async def execute(
        self,
        skill: SkillDefinition,
        arguments: dict[str, Any],
        parent_task_id: str,
    ) -> dict[str, Any]:
        """Execute a skill as a focused sub-conversation.

        Args:
            skill: The skill definition to execute.
            arguments: Input arguments matching the skill's input_schema.
            parent_task_id: Parent task ID for tracking.

        Returns:
            Dict with status and result text.
        """
        from agent_host.loop.agent_loop import AgentLoop

        task_id = f"{parent_task_id}-skill-{skill.name}"

        # Stage 2: load full content on demand (no-op for built-in skills)
        skill = SkillLoader.load_skill_content(skill)

        # Apply argument substitution to prompt content
        prompt_content = substitute_arguments(skill.prompt_content, arguments)

        # Build system prompt
        system_prompt = (
            f"You are executing the '{skill.name}' skill.\n\nSkill: {skill.description}\n\n"
        )
        if prompt_content:
            system_prompt += f"{prompt_content}\n\n"
        system_prompt += "Complete the task and report your results clearly."

        # Build user message from arguments
        user_message = self._format_arguments(skill, arguments)

        # Fresh thread for the skill
        thread = MessageThread(system_prompt=system_prompt)
        thread.add_user_message(user_message)

        compactor = DropOldestCompactor(recency_window=_SKILL_RECENCY_WINDOW)
        cancel = asyncio.Event()

        try:
            loop = AgentLoop(
                llm_client=self._llm_client,
                tool_executor=self._tool_executor,
                thread=thread,
                compactor=compactor,
                policy_enforcer=self._policy_enforcer,
                token_budget=self._token_budget,
                event_emitter=None,
                cancellation_event=cancel,
                max_steps=skill.max_steps,
                max_context_tokens=self._max_context_tokens,
            )

            result = await loop.run(task_id)

            result_text = result.text
            if len(result_text) > _RESULT_MAX_CHARS:
                result_text = result_text[:_RESULT_MAX_CHARS] + "... [truncated]"

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

    @staticmethod
    def _format_arguments(skill: SkillDefinition, arguments: dict[str, Any]) -> str:
        """Format skill arguments into a user message."""
        parts = [f"Execute the '{skill.name}' skill with these inputs:"]
        for key, value in arguments.items():
            parts.append(f"- {key}: {value}")
        return "\n".join(parts)
