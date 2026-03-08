"""Solo (no-op) strategy implementations.

These are the default strategies used when no coordination feature
(teams, MCP, etc.) is active. They preserve existing single-agent
behavior with zero overhead.
"""

from __future__ import annotations

from typing import Any


class SoloCoordinator:
    """No-op coordination — single agent, nothing to coordinate."""

    async def on_session_start(self, _session_id: str, _config: dict[str, Any]) -> None:
        pass

    async def on_session_shutdown(self) -> None:
        pass

    async def spawn_agent(
        self,
        _name: str,
        _role: str,
        _initial_prompt: str,
        _budget: int,
    ) -> dict[str, Any]:
        msg = "Cannot spawn agents without an active coordination strategy"
        raise RuntimeError(msg)

    async def shutdown_agent(self, _name: str) -> None:
        pass

    async def shutdown_all(self) -> None:
        pass

    def get_active_agents(self) -> list[dict[str, Any]]:
        return []


class SoloToolProvider:
    """No-op tool provider — no extra tools in solo mode."""

    def get_tool_definitions(self, _agent_role: str) -> list[dict[str, Any]]:
        return []

    async def handle_tool_call(
        self,
        tool_name: str,
        _arguments: dict[str, Any],
        _agent_name: str,
    ) -> dict[str, Any]:
        msg = f"SoloToolProvider does not handle tool: {tool_name}"
        raise RuntimeError(msg)

    def owns_tool(self, _tool_name: str) -> bool:
        return False


class SoloContextInjector:
    """No-op context injector — no extra context in solo mode."""

    async def get_injections(self, _agent_name: str) -> list[str]:
        return []

    def estimate_overhead_tokens(self) -> int:
        return 0


class SoloCheckpointProvider:
    """No-op checkpoint provider — no extra state to persist."""

    def capture(self) -> dict[str, Any]:
        return {}

    async def restore(self, _state: dict[str, Any]) -> None:
        pass
