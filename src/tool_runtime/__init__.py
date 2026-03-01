"""Local Tool Runtime — executes tool calls on the local desktop.

Public API:
    ToolRouter: Routes tool requests to registered tool implementations.
    ExecutionContext: Carries optional capability constraints from agent_host.
    ToolExecutionResult: Wraps ToolResult with optional artifact data.
    ArtifactData: Raw artifact bytes extracted from tool output.
"""

from tool_runtime.models import ArtifactData, ExecutionContext, ToolExecutionResult
from tool_runtime.router.tool_router import ToolRouter

__all__ = [
    "ArtifactData",
    "ExecutionContext",
    "ToolExecutionResult",
    "ToolRouter",
]
