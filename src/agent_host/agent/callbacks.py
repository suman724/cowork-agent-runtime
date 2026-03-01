"""ADK agent callbacks for policy enforcement, budget tracking, and events."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from agent_host.agent.tool_adapter import get_capability_for_tool
from agent_host.exceptions import PolicyExpiredError

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models import LlmRequest, LlmResponse  # type: ignore[attr-defined]

    from agent_host.budget.token_budget import TokenBudget
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.policy.policy_enforcer import PolicyEnforcer
    from agent_host.session.workspace_client import WorkspaceClient

logger = structlog.get_logger()


def make_before_tool_callback(
    policy_enforcer: PolicyEnforcer,
    event_emitter: EventEmitter | None = None,
) -> Any:
    """Create a before_tool_callback for policy enforcement.

    Returns None to proceed with tool execution, or a dict to skip
    the tool call (returned as the tool result to the LLM).
    """

    def before_tool_callback(
        callback_context: CallbackContext,  # noqa: ARG001
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        capability_name = get_capability_for_tool(tool_name)
        if not capability_name:
            return None

        result = policy_enforcer.check_tool_call(tool_name, capability_name, tool_input)

        if result.decision == "DENIED":
            logger.warning(
                "tool_call_denied",
                tool_name=tool_name,
                capability=capability_name,
                reason=result.reason,
            )
            return {
                "status": "denied",
                "error": {
                    "code": "CAPABILITY_DENIED",
                    "message": result.reason,
                },
            }

        if result.decision == "APPROVAL_REQUIRED":
            logger.info(
                "tool_call_requires_approval",
                tool_name=tool_name,
                capability=capability_name,
                risk_level=result.risk_level,
            )

        if event_emitter:
            event_emitter.emit_tool_requested(tool_name, capability_name, tool_input)

        return None

    return before_tool_callback


def make_before_model_callback(
    policy_enforcer: PolicyEnforcer,
    token_budget: TokenBudget,
) -> Any:
    """Create a before_model_callback for policy and budget enforcement.

    Returns None to proceed with the LLM call, or an LlmResponse to skip it.
    """

    def before_model_callback(
        callback_context: CallbackContext,
        llm_request: LlmRequest,  # noqa: ARG001
    ) -> LlmResponse | None:
        # Check policy (LLM.Call capability + not expired)
        result = policy_enforcer.check_llm_call()
        if result.decision == "DENIED":
            raise PolicyExpiredError(result.reason)

        # Check token budget
        token_budget.pre_check()

        # Track call start time in state for latency measurement
        callback_context.state["temp:llm_call_start"] = time.monotonic()

        return None

    return before_model_callback


def make_after_tool_callback(
    workspace_client: WorkspaceClient | None = None,  # noqa: ARG001
    event_emitter: EventEmitter | None = None,
) -> Any:
    """Create an after_tool_callback for event emission and artifact upload.

    Returns None to use the tool result as-is.
    """

    def after_tool_callback(
        callback_context: CallbackContext,  # noqa: ARG001
        tool_name: str,
        tool_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        status = tool_result.get("status", "unknown")

        if event_emitter:
            event_emitter.emit_tool_completed(tool_name, status)

        logger.debug(
            "tool_call_completed",
            tool_name=tool_name,
            status=status,
        )

        return None

    return after_tool_callback
