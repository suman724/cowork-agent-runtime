"""Tests for EventEmitter — structured event emission."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agent_host.events.event_emitter import EventEmitter
from agent_host.models import SessionContext


def _make_emitter(
    transport: Any = None,
) -> EventEmitter:
    ctx = SessionContext(
        session_id="sess-1",
        workspace_id="ws-1",
        tenant_id="t-1",
        user_id="u-1",
    )
    return EventEmitter(ctx, transport)


class TestEmitTaskFailed:
    def test_basic_reason_only(self) -> None:
        """emit_task_failed with just reason produces expected payload."""
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_task_failed("task-1", reason="Something went wrong")

            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args
            payload = call_kwargs.kwargs.get("payload") or call_kwargs[1].get("payload")
            assert payload["message"] == "Something went wrong"
            assert payload["isRecoverable"] is False
            assert "errorCode" not in payload
            assert "errorType" not in payload

    def test_default_message_when_no_reason(self) -> None:
        """emit_task_failed with no reason uses 'Task failed' as default."""
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_task_failed("task-1")

            payload = mock_emit.call_args.kwargs.get("payload") or mock_emit.call_args[1].get(
                "payload"
            )
            assert payload["message"] == "Task failed"

    def test_enriched_payload_with_error_code(self) -> None:
        """emit_task_failed includes errorCode, errorType, isRecoverable when provided."""
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_task_failed(
                "task-1",
                reason="Rate limited by the LLM provider.",
                error_code="RATE_LIMITED",
                error_type="rate_limit",
                is_recoverable=True,
            )

            payload = mock_emit.call_args.kwargs.get("payload") or mock_emit.call_args[1].get(
                "payload"
            )
            assert payload["message"] == "Rate limited by the LLM provider."
            assert payload["errorCode"] == "RATE_LIMITED"
            assert payload["errorType"] == "rate_limit"
            assert payload["isRecoverable"] is True

    def test_error_code_omitted_when_none(self) -> None:
        """errorCode and errorType are absent from payload when not provided."""
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_task_failed(
                "task-1",
                reason="Unknown error",
                is_recoverable=False,
            )

            payload = mock_emit.call_args.kwargs.get("payload") or mock_emit.call_args[1].get(
                "payload"
            )
            assert "errorCode" not in payload
            assert "errorType" not in payload
            assert payload["isRecoverable"] is False

    def test_severity_is_error(self) -> None:
        """emit_task_failed always emits with severity='error'."""
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_task_failed("task-1", reason="fail")

            call_kwargs = mock_emit.call_args
            severity = call_kwargs.kwargs.get("severity") or call_kwargs[1].get("severity")
            assert severity == "error"


class TestEmitSessionCreated:
    def test_emits_session_created(self) -> None:
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_session_created()
            mock_emit.assert_called_once()


class TestEmitSessionFailed:
    def test_emits_with_error_severity(self) -> None:
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_session_failed("Policy expired")

            call_kwargs = mock_emit.call_args
            severity = call_kwargs.kwargs.get("severity") or call_kwargs[1].get("severity")
            assert severity == "error"
            payload = call_kwargs.kwargs.get("payload") or call_kwargs[1].get("payload")
            assert payload["message"] == "Policy expired"


class TestEmitCheckpointSaved:
    def test_emits_with_step_number(self) -> None:
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_checkpoint_saved("task-1", 5)

            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args
            payload = call_kwargs.kwargs.get("payload") or call_kwargs[1].get("payload")
            assert payload["stepNumber"] == 5


class TestEmitCheckpointRestored:
    def test_emits_with_source(self) -> None:
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_checkpoint_restored(source="local")

            mock_emit.assert_called_once()
            call_kwargs = mock_emit.call_args
            payload = call_kwargs.kwargs.get("payload") or call_kwargs[1].get("payload")
            assert payload["source"] == "local"


class TestEmitCheckpointFailed:
    def test_emits_with_warning_severity(self) -> None:
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_checkpoint_failed("task-1", "disk full")

            call_kwargs = mock_emit.call_args
            severity = call_kwargs.kwargs.get("severity") or call_kwargs[1].get("severity")
            assert severity == "warning"
            payload = call_kwargs.kwargs.get("payload") or call_kwargs[1].get("payload")
            assert payload["reason"] == "disk full"


class TestEmitWorkspaceSyncCompleted:
    def test_emits_sync_completed(self) -> None:
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_workspace_sync_completed("task-1")
            mock_emit.assert_called_once()


class TestEmitWorkspaceSyncFailed:
    def test_emits_with_warning_severity(self) -> None:
        emitter = _make_emitter()
        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_workspace_sync_failed("task-1")

            call_kwargs = mock_emit.call_args
            severity = call_kwargs.kwargs.get("severity") or call_kwargs[1].get("severity")
            assert severity == "warning"
