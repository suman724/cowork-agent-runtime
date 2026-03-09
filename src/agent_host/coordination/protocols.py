"""Strategy protocols for extending agent behavior.

These protocols define extension points that allow new capabilities
(Agent Teams, MCP tools, RAG, etc.) to be injected into existing
components without modifying their core logic.

Each protocol has a Solo (no-op) implementation for non-extended sessions
and feature-specific implementations that add new behavior.

See: cowork-infra/docs/components/agent-teams.md §15 (Strategy-Based Decoupling)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CoordinationStrategy(Protocol):
    """How agents coordinate within a session.

    SessionManager delegates lifecycle and agent management to this strategy.
    Solo: no-ops (single agent, no coordination).
    Team: wraps TeamManager, SharedTaskList, MailboxRouter.
    """

    async def on_session_start(self, session_id: str, config: dict[str, Any]) -> None:
        """Called when the session starts. Set up coordination state."""
        ...

    async def on_session_shutdown(self) -> None:
        """Called during session shutdown. Clean up coordination state."""
        ...

    async def spawn_agent(
        self,
        name: str,
        role: str,
        initial_prompt: str,
        budget: int,
    ) -> dict[str, Any]:
        """Spawn a coordinated agent (teammate, specialist, etc.).

        Returns agent info dict with at minimum: {"name": str, "session_id": str}.
        """
        ...

    async def shutdown_agent(self, name: str) -> None:
        """Gracefully stop a coordinated agent."""
        ...

    async def shutdown_all(self) -> None:
        """Shut down all coordinated agents."""
        ...

    def get_active_agents(self) -> list[dict[str, Any]]:
        """List currently active coordinated agents."""
        ...


@runtime_checkable
class ToolProviderStrategy(Protocol):
    """Provides additional tools based on the agent's mode.

    AgentToolHandler asks this strategy for extra tool definitions
    and delegates handling of tools owned by the strategy.
    Solo: no extra tools.
    Team: team tools (CreateTeam, SendTeamMessage, etc.), filtered by agent role.
    """

    def get_tool_definitions(self, agent_role: str) -> list[dict[str, Any]]:
        """Return tool definitions for this agent's role.

        Args:
            agent_role: "lead", "teammate", or "solo".
        """
        ...

    async def handle_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_name: str,
    ) -> dict[str, Any]:
        """Handle a tool call for a tool provided by this strategy."""
        ...

    def owns_tool(self, tool_name: str) -> bool:
        """Return True if this strategy handles the given tool name."""
        ...


@runtime_checkable
class ContextInjectionStrategy(Protocol):
    """Injects additional context into the LLM message window.

    Called by LoopRuntime during context assembly. Injections are placed
    after working memory, before conversation history.
    Solo: no injections.
    Team: injects pending mailbox messages and current task status.
    """

    async def get_injections(self, agent_name: str) -> list[str]:
        """Return context strings to inject before the next LLM call.

        Each string becomes a system message inserted after working memory
        in the context assembly order.
        """
        ...

    def estimate_overhead_tokens(self) -> int:
        """Estimated token cost of injections (for compaction budget)."""
        ...

    def has_pending_messages(self, agent_name: str) -> bool:
        """Non-consuming check for pending messages.

        Used by teammates to decide whether to block or proceed
        with the next LLM call. Does not drain the message queue.
        """
        ...


@runtime_checkable
class CheckpointStrategy(Protocol):
    """Persists and restores additional state during checkpoints.

    CheckpointManager calls these to capture/restore strategy-specific
    state alongside the core session checkpoint.
    Solo: empty dict, restore is a no-op.
    Team: captures coordination state (member metadata, task list, mailbox queues).
    """

    def capture(self) -> dict[str, Any]:
        """Capture strategy-specific state as a serializable dict."""
        ...

    async def restore(self, state: dict[str, Any]) -> None:
        """Restore strategy state from a previously captured dict."""
        ...
