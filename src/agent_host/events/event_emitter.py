"""Event emitter — emits agent loop events as JSON-RPC notifications + structlog."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from cowork_platform_sdk import Component, EventType, build_event

if TYPE_CHECKING:
    from agent_host.models import SessionContext
    from agent_host.server.stdio_transport import StdioTransport

logger = structlog.get_logger()


class EventEmitter:
    """Emits events both as structured logs (stderr) and JSON-RPC notifications (stdout).

    All emission is fire-and-forget — errors are logged but never propagated.
    """

    def __init__(
        self,
        session_context: SessionContext,
        transport: StdioTransport | None = None,
    ) -> None:
        self._ctx = session_context
        self._transport = transport

    def emit(
        self,
        event_type: str,
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        """Emit a structured event.

        - Logs to stderr via structlog
        - Sends JSON-RPC notification to stdout (if transport available)
        """
        event = build_event(
            event_type=event_type,
            component=Component.LOCAL_AGENT_HOST,
            tenant_id=self._ctx.tenant_id,
            user_id=self._ctx.user_id,
            session_id=self._ctx.session_id,
            workspace_id=self._ctx.workspace_id,
            task_id=task_id,
            severity=severity,
            payload=payload or {},
        )

        # Log to stderr
        logger.info(
            "session_event",
            event_type=event_type,
            task_id=task_id,
            session_id=self._ctx.session_id,
        )

        # Send JSON-RPC notification (fire-and-forget)
        if self._transport:
            try:
                from agent_host.server.json_rpc import serialize_notification

                notification = serialize_notification("SessionEvent", event)
                self._transport.write_sync(notification)
            except Exception:
                logger.warning(
                    "event_notification_failed",
                    event_type=event_type,
                    exc_info=True,
                )

    def emit_session_created(self) -> None:
        """Emit session_created event."""
        self.emit(EventType.SESSION_CREATED)

    def emit_task_completed(self, task_id: str) -> None:
        """Emit task_completed event when a single task finishes successfully."""
        self.emit(EventType.TASK_COMPLETED, task_id=task_id)

    def emit_task_failed(self, task_id: str, reason: str | None = None) -> None:
        """Emit task_failed event when a single task fails."""
        payload = {"message": reason} if reason else {}
        self.emit(EventType.TASK_FAILED, task_id=task_id, payload=payload, severity="error")

    def emit_session_completed(self) -> None:
        """Emit session_completed event on clean session shutdown."""
        self.emit(EventType.SESSION_COMPLETED)

    def emit_session_failed(self, reason: str) -> None:
        """Emit session_failed event on session failure."""
        self.emit(EventType.SESSION_FAILED, payload={"message": reason}, severity="error")

    def emit_text_chunk(self, task_id: str, text: str) -> None:
        """Emit a text_chunk event (streaming LLM output)."""
        self.emit(
            EventType.TEXT_CHUNK,
            task_id=task_id,
            payload={"text": text},
        )

    def emit_tool_requested(
        self,
        tool_name: str,
        capability: str,
        arguments: dict[str, Any],
        tool_call_id: str = "",
        tool_type: str = "tool",
    ) -> None:
        """Emit tool_requested event."""
        self.emit(
            EventType.TOOL_REQUESTED,
            payload={
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "capability": capability,
                "arguments": arguments,
                "toolType": tool_type,
            },
        )

    def emit_tool_completed(
        self,
        tool_name: str,
        status: str,
        tool_call_id: str = "",
        result: str | None = None,
        error: str | None = None,
        tool_type: str = "tool",
    ) -> None:
        """Emit tool_completed event."""
        payload: dict[str, Any] = {
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "status": status,
            "toolType": tool_type,
        }
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error
        self.emit(EventType.TOOL_COMPLETED, payload=payload)

    def emit_approval_requested(
        self,
        approval_id: str,
        risk_level: str,
        tool_name: str,
        action_summary: str,
        session_id: str = "",
        task_id: str = "",
        title: str = "",
    ) -> None:
        """Emit approval_requested event.

        The Desktop's ``parseApprovalRequest`` requires ``sessionId``,
        ``taskId``, and ``title`` in the payload — otherwise it returns null.
        """
        self.emit(
            EventType.APPROVAL_REQUESTED,
            task_id=task_id or None,
            payload={
                "approvalId": approval_id,
                "sessionId": session_id or self._ctx.session_id,
                "taskId": task_id,
                "title": title or f"Approve {tool_name}",
                "riskLevel": risk_level,
                "toolName": tool_name,
                "actionSummary": action_summary,
            },
        )

    def emit_llm_retry(
        self,
        task_id: str,
        attempt: int,
        max_retries: int,
        error_message: str,
        delay_seconds: float,
    ) -> None:
        """Emit llm_retry event when retrying a transient LLM error."""
        self.emit(
            EventType.LLM_RETRY,
            task_id=task_id,
            payload={
                "attempt": attempt,
                "maxRetries": max_retries,
                "errorMessage": error_message,
                "delaySeconds": delay_seconds,
            },
            severity="warning",
        )

    def emit_step_limit_approaching(
        self,
        task_id: str,
        step_count: int,
        max_steps: int,
    ) -> None:
        """Emit step_limit_approaching event when nearing step budget."""
        self.emit(
            EventType.STEP_LIMIT_APPROACHING,
            task_id=task_id,
            payload={
                "currentStep": step_count,
                "maxSteps": max_steps,
            },
            severity="warning",
        )

    def emit_policy_expired(self) -> None:
        """Emit policy_expired event."""
        self.emit(EventType.POLICY_EXPIRED, severity="warning")

    def emit_step_started(self, task_id: str, step: int) -> None:
        """Emit step_started event at the beginning of each agent loop step."""
        self.emit(
            EventType.STEP_STARTED,
            task_id=task_id,
            payload={"stepNumber": step},
        )

    def emit_step_completed(self, task_id: str, step: int) -> None:
        """Emit step_completed event after each agent loop step finishes."""
        self.emit(
            EventType.STEP_COMPLETED,
            task_id=task_id,
            payload={"stepNumber": step},
        )

    def emit_context_compacted(
        self,
        task_id: str,
        messages_dropped: int,
        tokens_before: int,
        tokens_after: int,
    ) -> None:
        """Emit context_compacted event when message truncation occurs."""
        self.emit(
            EventType.CONTEXT_COMPACTED,
            task_id=task_id,
            payload={
                "messagesDropped": messages_dropped,
                "tokensBefore": tokens_before,
                "tokensAfter": tokens_after,
            },
        )
