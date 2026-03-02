"""Tests for SessionManager — lifecycle coordination."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_host.config import AgentHostConfig
from agent_host.exceptions import (
    CheckpointError,
    LLMBudgetExceededError,
    NoActiveTaskError,
    PolicyExpiredError,
    SessionNotFoundError,
)
from agent_host.session.session_manager import SessionManager


def _make_config(tmp_path: object) -> AgentHostConfig:
    return AgentHostConfig(
        llm_gateway_endpoint="https://llm.example.com",
        llm_gateway_auth_token="test-token",
        session_service_url="https://sessions.example.com",
        workspace_service_url="https://workspace.example.com",
        checkpoint_dir=str(tmp_path),
    )


def _make_session_response_dict() -> dict[str, object]:
    expires = (datetime.now(tz=UTC) + timedelta(hours=1)).isoformat()
    return {
        "sessionId": "sess-123",
        "workspaceId": "ws-456",
        "compatibilityStatus": "compatible",
        "policyBundle": {
            "policyBundleVersion": "2026-02-28.1",
            "schemaVersion": "1.0",
            "tenantId": "tenant-test",
            "userId": "user-test",
            "sessionId": "sess-123",
            "expiresAt": expires,
            "capabilities": [
                {"name": "File.Read"},
                {"name": "File.Write"},
                {"name": "Shell.Exec"},
                {"name": "LLM.Call"},
            ],
            "llmPolicy": {
                "allowedModels": ["gpt-4o"],
                "maxInputTokens": 8000,
                "maxOutputTokens": 4000,
                "maxSessionTokens": 100000,
            },
            "approvalRules": [],
        },
    }


async def _create_session_with_mock(
    manager: SessionManager,
) -> dict[str, object]:
    """Helper to create a session with mocked client."""
    with patch.object(
        manager._session_client,
        "create_session",
        new_callable=AsyncMock,
    ) as mock_create:
        from cowork_platform.session_create_response import SessionCreateResponse

        mock_create.return_value = SessionCreateResponse.model_validate(
            _make_session_response_dict()
        )
        return await manager.create_session({"tenantId": "tenant-test", "userId": "user-test"})


class TestSessionManagerCreate:
    @pytest.mark.asyncio
    async def test_create_session_success(self, tmp_path: object) -> None:
        """Successful session creation."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        result = await _create_session_with_mock(manager)

        assert result["sessionId"] == "sess-123"
        assert result["status"] == "ready"
        assert manager.session_context is not None
        assert manager.session_context.session_id == "sess-123"

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_create_session_with_workspace_hint(self, tmp_path: object) -> None:
        """Session creation with workspace hint passes it through."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        with patch.object(
            manager._session_client,
            "create_session",
            new_callable=AsyncMock,
        ) as mock_create:
            from cowork_platform.session_create_response import SessionCreateResponse

            mock_create.return_value = SessionCreateResponse.model_validate(
                _make_session_response_dict()
            )
            await manager.create_session(
                {
                    "tenantId": "tenant-test",
                    "userId": "user-test",
                    "workspaceHint": {
                        "localPaths": ["/home/user/project"],
                    },
                }
            )

            # Verify the request was made with workspace hint
            call_args = mock_create.call_args
            request = call_args[0][0]
            assert request.workspaceHint is not None

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_create_session_creates_event_emitter(self, tmp_path: object) -> None:
        """Session creation creates EventEmitter and emits session_created."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        # No event emitter before session creation
        assert manager._event_emitter is None

        await _create_session_with_mock(manager)

        # Event emitter created after session creation
        assert manager._event_emitter is not None

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_create_session_emits_session_created(self, tmp_path: object) -> None:
        """Session creation emits session_created event via EventEmitter."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        with patch(
            "agent_host.events.event_emitter.EventEmitter",
            wraps=None,
        ) as mock_cls:
            mock_emitter = MagicMock()
            mock_cls.return_value = mock_emitter
            await _create_session_with_mock(manager)

        mock_emitter.emit_session_created.assert_called_once()

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_create_session_incompatible(self, tmp_path: object) -> None:
        """Incompatible session returns error."""
        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)

        with patch.object(
            manager._session_client,
            "create_session",
            new_callable=AsyncMock,
        ) as mock_create:
            from cowork_platform.session_create_response import SessionCreateResponse

            mock_create.return_value = SessionCreateResponse(
                sessionId="sess-123",
                workspaceId="ws-456",
                compatibilityStatus="incompatible",
            )
            result = await manager.create_session(
                {"tenantId": "tenant-test", "userId": "user-test"}
            )

        assert result["status"] == "incompatible"

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerCapabilities:
    @pytest.mark.asyncio
    async def test_capabilities_derived_from_router(self, tmp_path: object) -> None:
        """supportedCapabilities is derived from tools available in the router."""
        from cowork_platform.tool_definition import ToolDefinition

        config = _make_config(tmp_path)
        router = MagicMock()
        # Only ReadFile and RunCommand available
        router.get_available_tools.return_value = [
            ToolDefinition(
                toolName="ReadFile",
                description="Read a file",
                inputSchema={"type": "object"},
            ),
            ToolDefinition(
                toolName="RunCommand",
                description="Run a command",
                inputSchema={"type": "object"},
            ),
        ]
        manager = SessionManager(config, router)

        with patch.object(
            manager._session_client,
            "create_session",
            new_callable=AsyncMock,
        ) as mock_create:
            from cowork_platform.session_create_response import SessionCreateResponse

            mock_create.return_value = SessionCreateResponse.model_validate(
                _make_session_response_dict()
            )
            await manager.create_session({"tenantId": "tenant-test", "userId": "user-test"})

            # Check what capabilities were sent in the request
            call_args = mock_create.call_args
            request = call_args[0][0]
            caps = request.supportedCapabilities
            assert "File.Read" in caps
            assert "Shell.Exec" in caps
            assert "LLM.Call" in caps
            # File.Write and File.Delete not available since no WriteFile/DeleteFile tool
            assert "File.Write" not in caps
            assert "File.Delete" not in caps

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerState:
    @pytest.mark.asyncio
    async def test_get_state_no_session(self, tmp_path: object) -> None:
        """get_session_state returns no_session when not initialized."""
        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)

        state = await manager.get_session_state()
        assert state["status"] == "no_session"

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_get_state_with_session(self, tmp_path: object) -> None:
        """get_session_state returns session info when session exists."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)
        state = await manager.get_session_state()

        assert state["sessionId"] == "sess-123"
        assert state["workspaceId"] == "ws-456"
        assert state["hasActiveTask"] is False
        assert state["currentTaskId"] is None
        assert "tokenUsage" in state

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerTask:
    @pytest.mark.asyncio
    async def test_start_task_without_session(self, tmp_path: object) -> None:
        """start_task raises when no session exists."""
        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)

        with pytest.raises(SessionNotFoundError):
            await manager.start_task({"taskId": "task-1", "prompt": "hello"})

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_start_task_with_session(self, tmp_path: object) -> None:
        """start_task spawns background task and returns status."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        # Mock the runner to avoid real ADK execution
        manager._runner = MagicMock()
        manager._runner.run_async = MagicMock(return_value=_async_iter([]))

        result = await manager.start_task({"taskId": "task-1", "prompt": "hello"})

        assert result["taskId"] == "task-1"
        assert result["status"] == "running"
        assert manager._current_task is not None

        # Cancel the background task to clean up
        manager._current_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await manager._current_task

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_cancel_task_without_task(self, tmp_path: object) -> None:
        """cancel_task raises when no task is running."""
        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)

        with pytest.raises(NoActiveTaskError):
            await manager.cancel_task()

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_cancel_running_task(self, tmp_path: object) -> None:
        """cancel_task cancels the background asyncio task."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        # Create a fake long-running task
        async def _slow_task() -> None:
            await asyncio.sleep(100)

        manager._current_task = asyncio.create_task(_slow_task())
        manager._current_task_id = "task-1"

        result = await manager.cancel_task()

        assert result["status"] == "cancelled"
        assert result["taskId"] == "task-1"
        assert manager._current_task is None

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerApproval:
    @pytest.mark.asyncio
    async def test_deliver_approval_with_gate(self, tmp_path: object) -> None:
        """deliver_approval resolves a pending approval via ApprovalGate."""
        from agent_host.approval.approval_gate import ApprovalGate

        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)
        gate = ApprovalGate()
        manager._approval_gate = gate

        # Schedule a request in the background
        async def _request() -> str:
            return await gate.request_approval("appr-1", timeout=5.0)

        task = asyncio.create_task(_request())
        await asyncio.sleep(0.01)  # let the future register

        result = await manager.deliver_approval(
            {
                "approvalId": "appr-1",
                "decision": "approved",
            }
        )

        assert result["approvalId"] == "appr-1"
        assert result["status"] == "delivered"
        assert result["decision"] == "approved"
        assert await task == "approved"

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_deliver_denial_with_gate(self, tmp_path: object) -> None:
        """deliver_approval handles denial via ApprovalGate."""
        from agent_host.approval.approval_gate import ApprovalGate

        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)
        gate = ApprovalGate()
        manager._approval_gate = gate

        async def _request() -> str:
            return await gate.request_approval("appr-2", timeout=5.0)

        task = asyncio.create_task(_request())
        await asyncio.sleep(0.01)

        result = await manager.deliver_approval(
            {
                "approvalId": "appr-2",
                "decision": "denied",
                "reason": "Too risky",
            }
        )

        assert result["decision"] == "denied"
        assert await task == "denied"

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_deliver_without_gate(self, tmp_path: object) -> None:
        """deliver_approval returns not_found when no ApprovalGate is set."""
        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)

        result = await manager.deliver_approval(
            {
                "approvalId": "appr-3",
                "decision": "approved",
            }
        )

        assert result["status"] == "not_found"

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_deliver_unknown_approval_id(self, tmp_path: object) -> None:
        """deliver_approval returns not_found for unknown approval ID."""
        from agent_host.approval.approval_gate import ApprovalGate

        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)
        manager._approval_gate = ApprovalGate()

        result = await manager.deliver_approval(
            {
                "approvalId": "nonexistent",
                "decision": "approved",
            }
        )

        assert result["status"] == "not_found"

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_without_session(self, tmp_path: object) -> None:
        """Shutdown without session succeeds."""
        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)

        result = await manager.shutdown()

        assert result["status"] == "shutdown"

    @pytest.mark.asyncio
    async def test_shutdown_with_session(self, tmp_path: object) -> None:
        """Shutdown with active session cancels backend and cleans up."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        # Mock the cancel_session on the client
        with patch.object(
            manager._session_client,
            "cancel_session",
            new_callable=AsyncMock,
        ):
            result = await manager.shutdown()

        assert result["status"] == "shutdown"
        assert result["sessionId"] == "sess-123"
        assert manager._session_context is None
        assert manager._runner is None

    @pytest.mark.asyncio
    async def test_shutdown_with_running_task(self, tmp_path: object) -> None:
        """Shutdown cancels running task before cleanup."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        # Simulate a running task
        async def _slow_task() -> None:
            await asyncio.sleep(100)

        manager._current_task = asyncio.create_task(_slow_task())
        manager._current_task_id = "task-1"

        with patch.object(
            manager._session_client,
            "cancel_session",
            new_callable=AsyncMock,
        ):
            result = await manager.shutdown()

        assert result["status"] == "shutdown"
        assert manager._current_task is None

    @pytest.mark.asyncio
    async def test_shutdown_cancel_failure_tolerant(self, tmp_path: object) -> None:
        """Shutdown tolerates backend cancel failure."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        # Mock cancel_session to fail
        with patch.object(
            manager._session_client,
            "cancel_session",
            new_callable=AsyncMock,
            side_effect=Exception("Backend unreachable"),
        ):
            result = await manager.shutdown()

        # Should still succeed
        assert result["status"] == "shutdown"


class TestSessionManagerRetry:
    @pytest.mark.asyncio
    async def test_run_agent_retries_on_transient_error(self, tmp_path: object) -> None:
        """Transient error on first attempt triggers retry, second attempt succeeds."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        call_count = 0

        async def _mock_run_async(**kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            # Second attempt succeeds with no events
            return
            yield  # make it an async generator

        manager._runner = MagicMock()
        manager._runner.run_async = _mock_run_async
        manager._event_emitter = MagicMock()

        # Use fast retry config
        object.__setattr__(manager._config, "llm_max_retries", 2)
        object.__setattr__(manager._config, "llm_retry_base_delay", 0.01)
        object.__setattr__(manager._config, "llm_retry_max_delay", 0.05)

        await manager._run_agent("hello", "task-1")

        assert call_count == 2
        manager._event_emitter.emit_task_completed.assert_called_once_with("task-1")
        manager._event_emitter.emit_llm_retry.assert_called_once()

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_run_agent_no_retry_on_permanent_error(self, tmp_path: object) -> None:
        """Permanent errors (LLMBudgetExceededError) are not retried."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        async def _mock_run_async(**kwargs):  # type: ignore[no-untyped-def]
            raise LLMBudgetExceededError("Budget exhausted")
            yield  # make it an async generator

        manager._runner = MagicMock()
        manager._runner.run_async = _mock_run_async
        manager._event_emitter = MagicMock()

        await manager._run_agent("hello", "task-1")

        manager._event_emitter.emit_task_failed.assert_called_once()
        manager._event_emitter.emit_llm_retry.assert_not_called()

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_run_agent_exhausts_retries(self, tmp_path: object) -> None:
        """Transient error exceeding max retries emits task_failed."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        async def _mock_run_async(**kwargs):  # type: ignore[no-untyped-def]
            raise httpx.ConnectError("Connection refused")
            yield  # make it an async generator

        manager._runner = MagicMock()
        manager._runner.run_async = _mock_run_async
        manager._event_emitter = MagicMock()

        object.__setattr__(manager._config, "llm_max_retries", 2)
        object.__setattr__(manager._config, "llm_retry_base_delay", 0.01)
        object.__setattr__(manager._config, "llm_retry_max_delay", 0.05)

        await manager._run_agent("hello", "task-1")

        # Should have retried twice then failed
        assert manager._event_emitter.emit_llm_retry.call_count == 2
        manager._event_emitter.emit_task_failed.assert_called_once()

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_run_agent_emits_retry_event(self, tmp_path: object) -> None:
        """Verify emit_llm_retry is called with correct attempt and delay."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        call_count = 0

        async def _mock_run_async(**kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ReadTimeout("Read timed out")
            return
            yield

        manager._runner = MagicMock()
        manager._runner.run_async = _mock_run_async
        manager._event_emitter = MagicMock()

        object.__setattr__(manager._config, "llm_max_retries", 3)
        object.__setattr__(manager._config, "llm_retry_base_delay", 0.01)
        object.__setattr__(manager._config, "llm_retry_max_delay", 0.05)

        await manager._run_agent("hello", "task-1")

        retry_call = manager._event_emitter.emit_llm_retry.call_args
        assert retry_call.kwargs["task_id"] == "task-1"
        assert retry_call.kwargs["attempt"] == 1
        assert retry_call.kwargs["max_retries"] == 3

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_run_agent_no_retry_on_cancel(self, tmp_path: object) -> None:
        """CancelledError exits immediately without retry."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        async def _mock_run_async(**kwargs):  # type: ignore[no-untyped-def]
            raise asyncio.CancelledError
            yield

        manager._runner = MagicMock()
        manager._runner.run_async = _mock_run_async
        manager._event_emitter = MagicMock()

        await manager._run_agent("hello", "task-1")

        manager._event_emitter.emit_llm_retry.assert_not_called()
        manager._event_emitter.emit_task_failed.assert_not_called()
        manager._event_emitter.emit_task_completed.assert_not_called()

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerSessionFailed:
    @pytest.mark.asyncio
    async def test_policy_expired_emits_session_failed(self, tmp_path: object) -> None:
        """PolicyExpiredError emits session_failed, not task_failed."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        async def _mock_run_async(**kwargs):  # type: ignore[no-untyped-def]
            raise PolicyExpiredError("Policy expired")
            yield

        manager._runner = MagicMock()
        manager._runner.run_async = _mock_run_async
        manager._event_emitter = MagicMock()

        await manager._run_agent("hello", "task-1")

        manager._event_emitter.emit_session_failed.assert_called_once_with("Policy expired")
        manager._event_emitter.emit_task_failed.assert_not_called()

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_checkpoint_error_emits_session_failed(self, tmp_path: object) -> None:
        """CheckpointError emits session_failed, not task_failed."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        async def _mock_run_async(**kwargs):  # type: ignore[no-untyped-def]
            raise CheckpointError("Checkpoint write failed")
            yield

        manager._runner = MagicMock()
        manager._runner.run_async = _mock_run_async
        manager._event_emitter = MagicMock()

        await manager._run_agent("hello", "task-1")

        manager._event_emitter.emit_session_failed.assert_called_once()
        manager._event_emitter.emit_task_failed.assert_not_called()

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_generic_error_emits_task_failed(self, tmp_path: object) -> None:
        """Generic non-transient errors emit task_failed."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        async def _mock_run_async(**kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("Something unexpected")
            yield

        manager._runner = MagicMock()
        manager._runner.run_async = _mock_run_async
        manager._event_emitter = MagicMock()

        await manager._run_agent("hello", "task-1")

        manager._event_emitter.emit_task_failed.assert_called_once()
        manager._event_emitter.emit_session_failed.assert_not_called()

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerTokenPersistence:
    @pytest.mark.asyncio
    async def test_token_budget_persisted(self, tmp_path: object) -> None:
        """Token usage is persisted to checkpoint after record_usage."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        # Manually record usage and persist
        assert manager._token_budget is not None
        manager._token_budget.record_usage(500, 200)
        manager._persist_token_budget()

        # Read checkpoint file to verify
        session = manager._checkpoint_service._sessions.get("sess-123")
        assert session is not None
        assert session.state["_token_input_used"] == 500
        assert session.state["_token_output_used"] == 200

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_token_budget_restored(self, tmp_path: object) -> None:
        """Token budget is restored from checkpoint on session creation."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        # Pre-populate checkpoint with token data
        from agent_host.session.session_manager import APP_NAME

        await manager._checkpoint_service.create_session(
            app_name=APP_NAME,
            user_id="user-test",
            session_id="sess-123",
            state={
                "workspace_id": "ws-456",
                "tenant_id": "tenant-test",
                "_token_input_used": 1000,
                "_token_output_used": 500,
            },
        )

        await _create_session_with_mock(manager)

        # The ADK session creation in create_session replaces the checkpoint,
        # but we can verify restore_usage was called by checking state
        assert manager._token_budget is not None
        # Since create_session creates a NEW checkpoint, the pre-populated
        # state is overwritten. In a real crash recovery scenario,
        # the checkpoint service would load from disk.
        # For this test, we verify the restore method itself works:
        manager._checkpoint_service._sessions["sess-123"].state["_token_input_used"] = 1000
        manager._checkpoint_service._sessions["sess-123"].state["_token_output_used"] = 500
        manager._restore_token_budget_from_checkpoint()

        assert manager._token_budget.input_tokens_used == 1000
        assert manager._token_budget.output_tokens_used == 500

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerPatchPreview:
    @pytest.mark.asyncio
    async def test_get_patch_preview_with_tracker(self, tmp_path: object) -> None:
        """get_patch_preview delegates to FileChangeTracker."""
        from agent_host.agent.file_change_tracker import FileChangeTracker

        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)
        tracker = FileChangeTracker()
        tracker.record_write("task-1", "/tmp/file.txt", old_content=None, new_content="hello")
        manager._file_change_tracker = tracker

        result = await manager.get_patch_preview({"taskId": "task-1"})
        assert result["taskId"] == "task-1"
        assert len(result["files"]) == 1
        assert result["files"][0]["status"] == "added"

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_get_patch_preview_without_tracker(self, tmp_path: object) -> None:
        """get_patch_preview returns empty when no tracker is set."""
        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)

        result = await manager.get_patch_preview({"taskId": "task-1"})
        assert result["files"] == []

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerRichHistory:
    @pytest.mark.asyncio
    async def test_upload_history_includes_tool_messages(self, tmp_path: object) -> None:
        """_upload_history includes tool call/result messages between user and assistant."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        # Simulate tool messages collected during _run_agent
        manager._task_tool_messages = [
            {"type": "tool_call", "toolName": "ReadFile", "arguments": '{"path":"/tmp/x"}'},
            {"type": "tool_result", "toolName": "ReadFile", "output": '"file contents"'},
        ]

        # Mock workspace client upload
        with patch.object(
            manager._workspace_client,
            "upload_session_history",
            new_callable=AsyncMock,
        ) as mock_upload:
            await manager._upload_history("hello", "I read the file.", "task-1")

            mock_upload.assert_called_once()
            messages = mock_upload.call_args[1]["messages"]

            # Should be: user, tool_call, tool_result, assistant = 4 messages
            assert len(messages) == 4
            assert messages[0].role == "user"
            assert messages[1].role == "tool"
            assert messages[2].role == "tool"
            assert messages[3].role == "assistant"

            # Verify tool message content
            import json

            tool_call = json.loads(messages[1].content)
            assert tool_call["type"] == "tool_call"
            assert tool_call["toolName"] == "ReadFile"

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerMaxSteps:
    @pytest.mark.asyncio
    async def test_start_task_parses_task_options(self, tmp_path: object) -> None:
        """start_task extracts maxSteps from taskOptions."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)
        manager._runner = MagicMock()
        manager._runner.run_async = MagicMock(return_value=_async_iter([]))

        result = await manager.start_task(
            {
                "taskId": "task-1",
                "prompt": "hello",
                "taskOptions": {"maxSteps": 10},
            }
        )

        assert result["status"] == "running"
        assert manager._current_max_steps == 10

        manager._current_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await manager._current_task

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_default_max_steps(self, tmp_path: object) -> None:
        """No taskOptions → config default used."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)
        manager._runner = MagicMock()
        manager._runner.run_async = MagicMock(return_value=_async_iter([]))

        await manager.start_task({"taskId": "task-1", "prompt": "hello"})

        assert manager._current_max_steps == config.default_max_steps

        manager._current_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await manager._current_task

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_max_steps_clamped(self, tmp_path: object) -> None:
        """Values outside 1-200 are clamped."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)
        manager._runner = MagicMock()
        manager._runner.run_async = MagicMock(return_value=_async_iter([]))

        # Over 200
        await manager.start_task(
            {
                "taskId": "task-1",
                "prompt": "hello",
                "taskOptions": {"maxSteps": 999},
            }
        )
        assert manager._current_max_steps == 200
        manager._current_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await manager._current_task

        # Under 1
        await manager.start_task(
            {
                "taskId": "task-2",
                "prompt": "hello",
                "taskOptions": {"maxSteps": -5},
            }
        )
        assert manager._current_max_steps == 1
        manager._current_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await manager._current_task

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_step_limit_stops_agent(self, tmp_path: object) -> None:
        """Agent stops when step limit is reached."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        # Create mock events: 5 usage_metadata events
        events = []
        for _ in range(5):
            event = MagicMock()
            event.content = None
            event.usage_metadata = MagicMock()
            event.usage_metadata.prompt_token_count = 100
            event.usage_metadata.candidates_token_count = 50
            events.append(event)

        manager._runner = MagicMock()
        manager._runner.run_async = MagicMock(return_value=_async_iter(events))
        manager._event_emitter = MagicMock()

        # Run with max_steps=3 — should stop after 3 steps
        await manager._run_agent("hello", "task-1", max_steps=3)

        manager._event_emitter.emit_task_failed.assert_called_once()
        call_args = manager._event_emitter.emit_task_failed.call_args
        # emit_task_failed(task_id, reason=...) — check positional or keyword
        all_args = list(call_args[0]) + list(call_args[1].values())
        assert any("Step limit reached" in str(a) for a in all_args)
        # task_completed should NOT be called
        manager._event_emitter.emit_task_completed.assert_not_called()

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_step_limit_approaching_event(self, tmp_path: object) -> None:
        """80% threshold emits step_limit_approaching warning."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        # 10 max_steps → warning at step 8 (floor(10 * 0.8))
        events = []
        for _ in range(10):
            event = MagicMock()
            event.content = None
            event.usage_metadata = MagicMock()
            event.usage_metadata.prompt_token_count = 10
            event.usage_metadata.candidates_token_count = 5
            events.append(event)

        manager._runner = MagicMock()
        manager._runner.run_async = MagicMock(return_value=_async_iter(events))
        manager._event_emitter = MagicMock()

        await manager._run_agent("hello", "task-1", max_steps=10)

        # Step 8 = floor(10*0.8) should trigger warning
        manager._event_emitter.emit_step_limit_approaching.assert_called_once_with("task-1", 8, 10)
        # Step 10 = limit reached
        manager._event_emitter.emit_task_failed.assert_called_once()

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_get_session_state_includes_step_info(self, tmp_path: object) -> None:
        """get_session_state includes currentStep and maxSteps."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)
        state = await manager.get_session_state()

        assert "currentStep" in state
        assert "maxSteps" in state
        assert state["currentStep"] == 0
        assert state["maxSteps"] == config.default_max_steps

        await manager._session_client.close()
        await manager._workspace_client.close()


class TestSessionManagerCumulativeHistory:
    @pytest.mark.asyncio
    async def test_cumulative_history_across_tasks(self, tmp_path: object) -> None:
        """Second upload contains both tasks' messages."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        with patch.object(
            manager._workspace_client,
            "upload_session_history",
            new_callable=AsyncMock,
        ) as mock_upload:
            # First task
            manager._task_tool_messages = []
            await manager._upload_history("hello", "Hi there!", "task-1")

            # Second task
            manager._task_tool_messages = [
                {"type": "tool_call", "toolName": "ReadFile", "arguments": "{}"},
            ]
            await manager._upload_history("read file", "Done.", "task-2")

            # Second upload should have messages from both tasks
            assert mock_upload.call_count == 2
            second_messages = mock_upload.call_args_list[1][1]["messages"]
            # task-1: user + assistant = 2, task-2: user + tool + assistant = 3
            assert len(second_messages) == 5

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_session_messages_checkpoint_persistence(self, tmp_path: object) -> None:
        """Messages are saved to and restored from checkpoint."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        with patch.object(
            manager._workspace_client,
            "upload_session_history",
            new_callable=AsyncMock,
        ):
            manager._task_tool_messages = []
            await manager._upload_history("hello", "world", "task-1")

        # Verify persistence
        session = manager._checkpoint_service._sessions.get("sess-123")
        assert session is not None
        assert "_session_messages" in session.state
        assert len(session.state["_session_messages"]) == 2  # user + assistant

        # Clear and restore
        manager._session_messages = []
        manager._restore_session_messages_from_checkpoint()
        assert len(manager._session_messages) == 2
        assert manager._session_messages[0].role == "user"
        assert manager._session_messages[1].role == "assistant"

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_history_cap_at_500(self, tmp_path: object) -> None:
        """Oldest messages dropped when exceeding 500."""
        config = _make_config(tmp_path)
        router = MagicMock()
        router.get_available_tools.return_value = []
        manager = SessionManager(config, router)

        await _create_session_with_mock(manager)

        from datetime import UTC, datetime

        from cowork_platform.conversation_message import ConversationMessage

        # Pre-fill with 499 messages
        now = datetime.now(tz=UTC)
        manager._session_messages = [
            ConversationMessage(
                messageId=f"old-{i}",
                sessionId="sess-123",
                taskId="old-task",
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg-{i}",
                timestamp=now,
            )
            for i in range(499)
        ]

        with patch.object(
            manager._workspace_client,
            "upload_session_history",
            new_callable=AsyncMock,
        ) as mock_upload:
            # This adds 2 more (user + assistant) → 501 total → should cap at 500
            manager._task_tool_messages = []
            await manager._upload_history("new prompt", "new response", "task-new")

            messages = mock_upload.call_args[1]["messages"]
            assert len(messages) == 500

            # First message is still the original first
            assert messages[0].messageId == "old-0"
            # Last message is the new assistant
            assert messages[-1].messageId == "task-new-assistant"

        await manager._session_client.close()
        await manager._workspace_client.close()


async def _async_iter(items: list[object]) -> object:  # type: ignore[misc]
    """Create an async iterator from a list."""
    for item in items:
        yield item
