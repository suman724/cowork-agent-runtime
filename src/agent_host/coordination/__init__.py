"""Strategy interfaces for extending agent behavior.

Strategies decouple optional capabilities (teams, MCP tools, RAG, etc.)
from core components. Each strategy has a Solo (no-op) implementation
that preserves existing behavior, and feature-specific implementations
that add new functionality without modifying core code.
"""

from agent_host.coordination.protocols import (
    CheckpointStrategy,
    ContextInjectionStrategy,
    CoordinationStrategy,
    ToolProviderStrategy,
)
from agent_host.coordination.solo import (
    SoloCheckpointProvider,
    SoloContextInjector,
    SoloCoordinator,
    SoloToolProvider,
)

__all__ = [
    "CheckpointStrategy",
    "ContextInjectionStrategy",
    "CoordinationStrategy",
    "SoloCheckpointProvider",
    "SoloContextInjector",
    "SoloCoordinator",
    "SoloToolProvider",
    "ToolProviderStrategy",
]
