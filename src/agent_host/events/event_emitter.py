"""Event emitter — emits agent loop events as JSON-RPC notifications + structlog.

All events are buffered in an ``EventBuffer`` for replay by any transport
(SSE reconnection for HttpTransport, ``GetEvents`` RPC for Desktop App).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from cowork_platform_sdk import Component, EventType, build_event

from agent_host.events.event_buffer import EventBuffer

if TYPE_CHECKING:
    from agent_sdk.models import SessionContext

    from agent_host.transport.transport import Transport

logger = structlog.get_logger()


class EventEmitter:
    """Emits events as structured logs (stderr), to the transport, and to an event buffer.

    The event buffer enables replay for:
    - HttpTransport: SSE ``/events?since={id}`` for reconnecting web clients
    - StdioTransport: ``GetEvents`` JSON-RPC method for Desktop App view navigation

    All emission is fire-and-forget — errors are logged but never propagated.
    """

    def __init__(
        self,
        session_context: SessionContext,
        transport: Transport | None = None,
        event_buffer: EventBuffer | None = None,
    ) -> None:
        self._ctx = session_context
        self._transport = transport
        self._event_buffer = event_buffer or EventBuffer()

    @property
    def event_buffer(self) -> EventBuffer:
        """Access the shared event buffer."""
        return self._event_buffer

    def emit(
        self,
        event_type: str,
        task_id: str | None = None,
        step_id: str | None = None,
        payload: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        """Emit a structured event.

        1. Pushes to the event buffer (assigns monotonic ID)
        2. Logs to stderr via structlog
        3. Sends event via transport with eventId (if transport available)
        """
        event = build_event(
            event_type=event_type,
            component=Component.LOCAL_AGENT_HOST,
            tenant_id=self._ctx.tenant_id,
            user_id=self._ctx.user_id,
            session_id=self._ctx.session_id,
            workspace_id=self._ctx.workspace_id,
            task_id=task_id,
            step_id=step_id,
            severity=severity,
            payload=payload or {},
        )

        # 1. Buffer for replay (always, regardless of transport)
        event_id = self._event_buffer.push(event)

        # 2. Log to stderr
        logger.info(
            "session_event",
            event_type=event_type,
            event_id=event_id,
            task_id=task_id,
            session_id=self._ctx.session_id,
        )

        # 3. Send via transport with eventId (fire-and-forget)
        if self._transport:
            try:
                event_with_id = {**event, "eventId": event_id}
                self._transport.send_event(event_with_id)
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

    def emit_task_failed(
        self,
        task_id: str,
        reason: str | None = None,
        *,
        error_code: str | None = None,
        error_type: str | None = None,
        is_recoverable: bool = False,
    ) -> None:
        """Emit task_failed event when a single task fails."""
        payload: dict[str, Any] = {"message": reason or "Task failed"}
        if error_code:
            payload["errorCode"] = error_code
        if error_type:
            payload["errorType"] = error_type
        payload["isRecoverable"] = is_recoverable
        self.emit(EventType.TASK_FAILED, task_id=task_id, payload=payload, severity="error")

    def emit_session_completed(self) -> None:
        """Emit session_completed event on clean session shutdown."""
        self.emit(EventType.SESSION_COMPLETED)

    def emit_session_failed(self, reason: str) -> None:
        """Emit session_failed event on session failure."""
        self.emit(EventType.SESSION_FAILED, payload={"message": reason}, severity="error")

    def emit_task_started(self, task_id: str, prompt: str = "") -> None:
        """Emit task_started event when a new task begins."""
        payload: dict[str, Any] = {}
        if prompt:
            payload["prompt"] = prompt[:200]
        self.emit(EventType.TASK_STARTED, task_id=task_id, payload=payload)

    def emit_text_chunk(self, task_id: str, text: str, step_id: str | None = None) -> None:
        """Emit a text_chunk event (streaming LLM output)."""
        self.emit(
            EventType.TEXT_CHUNK,
            task_id=task_id,
            step_id=step_id,
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

    def emit_tool_output_chunk(
        self,
        tool_name: str,
        tool_call_id: str,
        content: str,
        task_id: str = "",
    ) -> None:
        """Emit tool_output_chunk event for streaming tool output to frontend."""
        self.emit(
            "tool_output_chunk",
            task_id=task_id,
            payload={
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "content": content,
            },
        )

    # --- Browser events ---

    def emit_browser_started(self, browser_channel: str = "chromium") -> None:
        """Emit browser_started event — browser launched, side panel should open."""
        self.emit("browser_started", payload={"browserChannel": browser_channel})

    def emit_browser_stopped(self, reason: str) -> None:
        """Emit browser_stopped event — browser closed (idle, closed, crashed)."""
        self.emit("browser_stopped", payload={"reason": reason})

    def emit_browser_page_state(self, url: str, screenshot_base64: str) -> None:
        """Emit browser_page_state event — screenshot update for side panel."""
        self.emit(
            "browser_page_state",
            payload={"url": url, "screenshotBase64": screenshot_base64},
        )

    def emit_browser_auth_required(self, domain: str, signals: list[str]) -> None:
        """Emit browser_auth_required — user needs to log in."""
        self.emit(
            "browser_auth_required",
            payload={"domain": domain, "signals": signals},
        )

    def emit_browser_takeover_started(self) -> None:
        """Emit browser_takeover_started — user took over browser control."""
        self.emit("browser_takeover_started")

    def emit_browser_takeover_ended(self) -> None:
        """Emit browser_takeover_ended — user resumed agent control."""
        self.emit("browser_takeover_ended")

    def emit_browser_domain_approved(self, domain: str) -> None:
        """Emit browser_domain_approved — user approved a new domain."""
        self.emit("browser_domain_approved", payload={"domain": domain})

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

    def emit_step_started(self, task_id: str, step: int, step_id: str | None = None) -> None:
        """Emit step_started event at the beginning of each agent loop step."""
        self.emit(
            EventType.STEP_STARTED,
            task_id=task_id,
            step_id=step_id,
            payload={"stepNumber": step},
        )

    def emit_step_completed(self, task_id: str, step: int, step_id: str | None = None) -> None:
        """Emit step_completed event after each agent loop step finishes."""
        self.emit(
            EventType.STEP_COMPLETED,
            task_id=task_id,
            step_id=step_id,
            payload={"stepNumber": step},
        )

    def emit_context_compacted(
        self,
        task_id: str,
        messages_dropped: int,
        tokens_before: int,
        tokens_after: int,
        step_id: str | None = None,
    ) -> None:
        """Emit context_compacted event when message truncation occurs."""
        self.emit(
            EventType.CONTEXT_COMPACTED,
            task_id=task_id,
            step_id=step_id,
            payload={
                "messagesDropped": messages_dropped,
                "tokensBefore": tokens_before,
                "tokensAfter": tokens_after,
            },
        )

    def emit_checkpoint_saved(self, task_id: str, step: int, step_id: str | None = None) -> None:
        """Emit checkpoint_saved event after each per-step checkpoint write."""
        self.emit(
            EventType.CHECKPOINT_SAVED,
            task_id=task_id,
            step_id=step_id,
            payload={"stepNumber": step},
        )

    def emit_checkpoint_restored(self, source: str = "local") -> None:
        """Emit checkpoint_restored event after state restoration."""
        self.emit(
            EventType.CHECKPOINT_RESTORED,
            payload={"source": source},
        )

    def emit_checkpoint_failed(self, task_id: str, reason: str) -> None:
        """Emit checkpoint_failed event when checkpoint write fails."""
        self.emit(
            EventType.CHECKPOINT_FAILED,
            task_id=task_id,
            payload={"reason": reason},
            severity="warning",
        )

    def emit_workspace_sync_completed(self, task_id: str) -> None:
        """Emit workspace_sync_completed event after periodic sync succeeds."""
        self.emit(
            EventType.WORKSPACE_SYNC_COMPLETED,
            task_id=task_id,
        )

    def emit_workspace_sync_failed(self, task_id: str) -> None:
        """Emit workspace_sync_failed event when periodic sync fails."""
        self.emit(
            EventType.WORKSPACE_SYNC_FAILED,
            task_id=task_id,
            severity="warning",
        )

    def emit_plan_mode_changed(
        self,
        task_id: str,
        plan_mode: bool,
        source: str = "agent",
    ) -> None:
        """Emit plan_mode_changed event when plan mode state changes."""
        self.emit(
            EventType.PLAN_MODE_CHANGED,
            task_id=task_id,
            payload={
                "planMode": plan_mode,
                "source": source,
            },
        )

    def emit_plan_updated(
        self,
        task_id: str,
        goal: str,
        steps: list[dict[str, Any]],
    ) -> None:
        """Emit plan_updated event when the agent creates or updates a plan."""
        self.emit(
            EventType.PLAN_UPDATED,
            task_id=task_id,
            payload={
                "goal": goal,
                "steps": steps,
            },
        )

    def emit_verification_started(self, task_id: str) -> None:
        """Emit verification_started event when verification phase begins."""
        self.emit(EventType.VERIFICATION_STARTED, task_id=task_id)

    def emit_verification_completed(
        self,
        task_id: str,
        *,
        passed: bool = True,
    ) -> None:
        """Emit verification_completed event when verification phase ends."""
        self.emit(
            EventType.VERIFICATION_COMPLETED,
            task_id=task_id,
            payload={"passed": passed},
        )
