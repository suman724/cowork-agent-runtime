"""Local Agent Host — orchestrates the agent loop on the user's desktop.

Public API:
    AgentHostConfig: Configuration from environment variables.
    SessionContext: Immutable session identity container (re-export from agent_sdk).
    PolicyCheckResult: Result of a policy capability check (re-export from agent_sdk).
    AgentHostError: Base exception hierarchy (re-export from agent_sdk).
"""

from agent_sdk.exceptions import AgentHostError
from agent_sdk.models import PolicyCheckResult, SessionContext

from agent_host.config import AgentHostConfig

__all__ = [
    "AgentHostConfig",
    "AgentHostError",
    "PolicyCheckResult",
    "SessionContext",
]
