"""LoopStrategy protocol — pluggable agent loop orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_host.loop.models import LoopResult


@runtime_checkable
class LoopStrategy(Protocol):
    """A pluggable agent loop orchestration strategy.

    Receives a LoopRuntime and makes all decisions about:
    - How to assemble context (messages) for each LLM call
    - When and how many times to call the LLM
    - When and how to execute tools
    - When to inject memory, working memory, error recovery prompts
    - When to spawn sub-agents or execute skills
    - When to terminate (completion, step limit, cancellation)
    """

    async def run(self, task_id: str) -> LoopResult: ...
