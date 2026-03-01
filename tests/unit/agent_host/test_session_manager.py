"""Tests for SessionManager — lifecycle coordination."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_host.config import AgentHostConfig
from agent_host.exceptions import NoActiveTaskError, SessionNotFoundError
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
    async def test_deliver_approval(self, tmp_path: object) -> None:
        """deliver_approval returns status."""
        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)

        result = await manager.deliver_approval(
            {
                "approvalId": "appr-1",
                "decision": "approved",
            }
        )

        assert result["approvalId"] == "appr-1"
        assert result["status"] == "delivered"
        assert result["decision"] == "approved"

        await manager._session_client.close()
        await manager._workspace_client.close()

    @pytest.mark.asyncio
    async def test_deliver_denial(self, tmp_path: object) -> None:
        """deliver_approval handles denial."""
        config = _make_config(tmp_path)
        router = MagicMock()
        manager = SessionManager(config, router)

        result = await manager.deliver_approval(
            {
                "approvalId": "appr-2",
                "decision": "denied",
                "reason": "Too risky",
            }
        )

        assert result["decision"] == "denied"

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


async def _async_iter(items: list[object]) -> object:  # type: ignore[misc]
    """Create an async iterator from a list."""
    for item in items:
        yield item
