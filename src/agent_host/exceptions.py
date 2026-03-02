"""Agent Host exception hierarchy mapped to JSON-RPC error codes."""

from __future__ import annotations


class AgentHostError(Exception):
    """Base exception for all Agent Host errors.

    Each subclass carries a JSON-RPC error code so the server layer
    can translate it into a proper JSON-RPC error response.
    """

    json_rpc_code: int = -32000

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


# --- Session errors ---


class SessionNotFoundError(AgentHostError):
    """No active session with the given ID."""

    json_rpc_code = -32001


class SessionExpiredError(AgentHostError):
    """Session has expired."""

    json_rpc_code = -32002


class PolicyExpiredError(AgentHostError):
    """Policy bundle has expired."""

    json_rpc_code = -32003


# --- LLM errors ---


class LLMGatewayError(AgentHostError):
    """LLM Gateway returned an unrecoverable error."""

    json_rpc_code = -32010


class LLMBudgetExceededError(AgentHostError):
    """Session token budget exhausted."""

    json_rpc_code = -32011


class LLMGuardrailBlockedError(AgentHostError):
    """LLM guardrail blocked the request."""

    json_rpc_code = -32012


# --- Policy errors ---


class CapabilityDeniedError(AgentHostError):
    """Policy denies the requested capability."""

    json_rpc_code = -32020


class ApprovalRequiredError(AgentHostError):
    """Tool call requires user approval before execution."""

    json_rpc_code = -32021

    def __init__(
        self,
        message: str,
        *,
        approval_rule_id: str,
        risk_level: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.approval_rule_id = approval_rule_id
        self.risk_level = risk_level


# --- Other errors ---


class ApprovalTimeoutError(AgentHostError):
    """User did not respond to approval request within the timeout."""

    json_rpc_code = -32022


class CheckpointError(AgentHostError):
    """Checkpoint read/write failure."""

    json_rpc_code = -32030


class TaskCancelledError(AgentHostError):
    """Task was cancelled by user request."""

    json_rpc_code = -32040


class NoActiveTaskError(AgentHostError):
    """No task is currently running."""

    json_rpc_code = -32041


class LoopAbortedError(AgentHostError):
    """Agent loop was aborted due to an unrecoverable error."""

    json_rpc_code = -32050
