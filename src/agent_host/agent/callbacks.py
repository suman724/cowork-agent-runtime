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

    from agent_host.agent.artifact_store import PendingArtifactStore
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

    Note: APPROVAL_REQUIRED is handled inside tool_fn (via ApprovalGate),
    not here, because before_tool_callback is synchronous and cannot await.
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

        # APPROVAL_REQUIRED: let tool_fn handle via ApprovalGate (it can await).
        # Just emit tool_requested for both ALLOWED and APPROVAL_REQUIRED.
        if event_emitter:
            event_emitter.emit_tool_requested(tool_name, capability_name, tool_input)

        return None

    return before_tool_callback


def make_before_model_callback(
    policy_enforcer: PolicyEnforcer,
    token_budget: TokenBudget,
    event_emitter: EventEmitter | None = None,
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
            if event_emitter:
                event_emitter.emit_policy_expired()
            raise PolicyExpiredError(result.reason)

        # Check token budget
        token_budget.pre_check()

        # Track call start time in state for latency measurement
        callback_context.state["temp:llm_call_start"] = time.monotonic()

        return None

    return before_model_callback


def make_after_tool_callback(
    workspace_client: WorkspaceClient | None = None,
    event_emitter: EventEmitter | None = None,
    artifact_store: PendingArtifactStore | None = None,
    session_id: str = "",
    workspace_id: str = "",
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

        # Upload artifacts (best-effort, fire-and-forget)
        if artifact_store and workspace_client and workspace_id:
            artifacts = artifact_store.pop(tool_name)
            for artifact in artifacts:
                try:
                    import asyncio

                    _task = asyncio.create_task(  # noqa: RUF006
                        workspace_client.upload_artifact(
                            workspace_id=workspace_id,
                            session_id=session_id,
                            artifact_data=artifact.data,
                            artifact_type=artifact.artifact_type,
                            artifact_name=artifact.artifact_name,
                            content_type=artifact.media_type,
                        )
                    )
                except Exception:
                    logger.warning(
                        "artifact_upload_failed",
                        tool_name=tool_name,
                        artifact_name=artifact.artifact_name,
                        exc_info=True,
                    )

        logger.debug(
            "tool_call_completed",
            tool_name=tool_name,
            status=status,
        )

        return None

    return after_tool_callback
