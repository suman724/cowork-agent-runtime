"""AgentLoop — deprecated alias for ReactLoop.

Use ``from agent_host.loop.react_loop import ReactLoop`` instead.
This module exists only for backward compatibility and will be removed.
"""

from agent_host.loop.react_loop import ReactLoop as AgentLoop

__all__ = ["AgentLoop"]
