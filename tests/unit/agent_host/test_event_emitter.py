"""Tests for EventEmitter — event bridging to JSON-RPC notifications."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cowork_platform_sdk import EventType

from agent_host.events.event_emitter import EventEmitter
from agent_host.models import SessionContext


def _make_context() -> SessionContext:
    return SessionContext(
        session_id="sess-123",
        workspace_id="ws-456",
        tenant_id="tenant-test",
        user_id="user-test",
    )


class TestEventEmitter:
    def test_emit_creates_event(self) -> None:
        """emit() creates a structured event via build_event."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch("agent_host.events.event_emitter.build_event") as mock_build:
            mock_build.return_value = {"eventType": "session_created"}
            emitter.emit(EventType.SESSION_CREATED)
            mock_build.assert_called_once()

    def test_emit_session_created(self) -> None:
        """emit_session_created calls emit with correct type."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_session_created()
            mock_emit.assert_called_once_with(EventType.SESSION_CREATED)

    def test_emit_text_chunk(self) -> None:
        """emit_text_chunk includes text in payload."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_text_chunk("task-1", "hello world")
            mock_emit.assert_called_once_with(
                EventType.TEXT_CHUNK,
                task_id="task-1",
                payload={"text": "hello world"},
            )

    def test_emit_tool_requested(self) -> None:
        """emit_tool_requested includes toolCallId and arguments."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_tool_requested(
                "ReadFile", "File.Read", {"path": "/tmp/x"}, tool_call_id="call_abc"
            )
            mock_emit.assert_called_once()
            payload = mock_emit.call_args[1]["payload"]
            assert payload["toolCallId"] == "call_abc"
            assert payload["toolName"] == "ReadFile"
            assert payload["arguments"] == {"path": "/tmp/x"}
            assert payload["capability"] == "File.Read"

    def test_emit_tool_completed(self) -> None:
        """emit_tool_completed includes toolCallId, result, and error."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_tool_completed(
                "ReadFile", "succeeded", tool_call_id="call_abc", result="file contents"
            )
            mock_emit.assert_called_once()
            payload = mock_emit.call_args[1]["payload"]
            assert payload["toolCallId"] == "call_abc"
            assert payload["status"] == "succeeded"
            assert payload["result"] == "file contents"

    def test_emit_tool_completed_with_error(self) -> None:
        """emit_tool_completed includes error field when present."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_tool_completed(
                "ReadFile", "failed", tool_call_id="call_abc", error="file not found"
            )
            payload = mock_emit.call_args[1]["payload"]
            assert payload["error"] == "file not found"

    def test_emit_tool_completed_omits_null_result_and_error(self) -> None:
        """emit_tool_completed omits result/error when None."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_tool_completed("ReadFile", "succeeded", tool_call_id="call_abc")
            payload = mock_emit.call_args[1]["payload"]
            assert "result" not in payload
            assert "error" not in payload

    def test_emit_with_transport(self) -> None:
        """emit() sends JSON-RPC notification when transport available."""
        ctx = _make_context()
        transport = MagicMock()
        transport.write_sync = MagicMock()
        emitter = EventEmitter(ctx, transport=transport)

        with patch("agent_host.events.event_emitter.build_event") as mock_build:
            mock_build.return_value = {"eventType": "session_created"}
            # Patch the lazy import of serialize_notification
            mock_serialize = MagicMock(return_value='{"jsonrpc":"2.0"}')
            mock_json_rpc = MagicMock()
            mock_json_rpc.serialize_notification = mock_serialize
            with patch.dict(
                "sys.modules",
                {"agent_host.server.json_rpc": mock_json_rpc},
            ):
                emitter.emit(EventType.SESSION_CREATED)
                transport.write_sync.assert_called_once()

    def test_emit_without_transport_no_error(self) -> None:
        """emit() works without transport (no notification sent)."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch("agent_host.events.event_emitter.build_event") as mock_build:
            mock_build.return_value = {"eventType": "session_created"}
            # Should not raise
            emitter.emit(EventType.SESSION_CREATED)

    def test_emit_transport_error_logged(self) -> None:
        """Transport errors are logged but don't propagate."""
        ctx = _make_context()
        transport = MagicMock()
        transport.write_sync = MagicMock(side_effect=OSError("broken pipe"))
        emitter = EventEmitter(ctx, transport=transport)

        with patch("agent_host.events.event_emitter.build_event") as mock_build:
            mock_build.return_value = {"eventType": "session_created"}
            mock_json_rpc = MagicMock()
            mock_json_rpc.serialize_notification = MagicMock(return_value='{"jsonrpc":"2.0"}')
            with patch.dict(
                "sys.modules",
                {"agent_host.server.json_rpc": mock_json_rpc},
            ):
                # Should not raise
                emitter.emit(EventType.SESSION_CREATED)

    def test_emit_approval_requested(self) -> None:
        """emit_approval_requested includes all required fields for Desktop parsing."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_approval_requested(
                "appr-1",
                "medium",
                "RunCommand",
                "Run: pytest",
                session_id="sess-123",
                task_id="task-1",
                title="Approve RunCommand",
            )
            mock_emit.assert_called_once()
            payload = mock_emit.call_args[1]["payload"]
            assert payload["approvalId"] == "appr-1"
            assert payload["riskLevel"] == "medium"
            assert payload["sessionId"] == "sess-123"
            assert payload["taskId"] == "task-1"
            assert payload["title"] == "Approve RunCommand"

    def test_emit_approval_requested_defaults(self) -> None:
        """emit_approval_requested falls back to context session_id and default title."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_approval_requested("appr-2", "high", "DeleteFile", "Delete /tmp/x")
            mock_emit.assert_called_once()
            payload = mock_emit.call_args[1]["payload"]
            assert payload["sessionId"] == "sess-123"  # from context
            assert payload["title"] == "Approve DeleteFile"  # default

    def test_emit_step_started_uses_step_number(self) -> None:
        """emit_step_started uses stepNumber key (not step) for desktop compat."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_step_started("task-1", 3)
            payload = mock_emit.call_args[1]["payload"]
            assert payload["stepNumber"] == 3
            assert "step" not in payload

    def test_emit_step_completed_uses_step_number(self) -> None:
        """emit_step_completed uses stepNumber key for desktop compat."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_step_completed("task-1", 5)
            payload = mock_emit.call_args[1]["payload"]
            assert payload["stepNumber"] == 5
            assert "step" not in payload

    def test_emit_policy_expired(self) -> None:
        """emit_policy_expired emits with warning severity."""
        ctx = _make_context()
        emitter = EventEmitter(ctx)

        with patch.object(emitter, "emit") as mock_emit:
            emitter.emit_policy_expired()
            mock_emit.assert_called_once_with(EventType.POLICY_EXPIRED, severity="warning")
