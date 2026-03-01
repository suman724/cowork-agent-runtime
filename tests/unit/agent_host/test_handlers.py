"""Tests for JSON-RPC handlers — thin delegation with mocked dependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_host.server.handlers import Handlers
from agent_host.server.method_dispatcher import MethodDispatcher


@pytest.fixture
def mock_session_manager() -> MagicMock:
    """Create a mock SessionManager."""
    manager = MagicMock()
    manager.create_session = AsyncMock(return_value={"sessionId": "sess-1", "status": "ready"})
    manager.start_task = AsyncMock(return_value={"taskId": "task-1", "status": "running"})
    manager.cancel_task = AsyncMock(return_value={"taskId": "task-1", "status": "cancelled"})
    manager.get_session_state = AsyncMock(return_value={"sessionId": "sess-1", "status": "running"})
    manager.deliver_approval = AsyncMock(
        return_value={"approvalId": "appr-1", "status": "delivered"}
    )
    manager.shutdown = AsyncMock(return_value={"status": "shutdown"})
    return manager


class TestHandlers:
    @pytest.mark.asyncio
    async def test_create_session(self, mock_session_manager: MagicMock) -> None:
        handlers = Handlers(mock_session_manager)
        result = await handlers.handle_create_session({"tenantId": "t", "userId": "u"})
        assert result["sessionId"] == "sess-1"
        mock_session_manager.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_task(self, mock_session_manager: MagicMock) -> None:
        handlers = Handlers(mock_session_manager)
        result = await handlers.handle_start_task({"taskId": "task-1", "prompt": "hello"})
        assert result["taskId"] == "task-1"
        mock_session_manager.start_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_task(self, mock_session_manager: MagicMock) -> None:
        handlers = Handlers(mock_session_manager)
        result = await handlers.handle_cancel_task({})
        assert result["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_get_session_state(self, mock_session_manager: MagicMock) -> None:
        handlers = Handlers(mock_session_manager)
        result = await handlers.handle_get_session_state({})
        assert result["sessionId"] == "sess-1"

    @pytest.mark.asyncio
    async def test_approve_action(self, mock_session_manager: MagicMock) -> None:
        handlers = Handlers(mock_session_manager)
        result = await handlers.handle_approve_action(
            {"approvalId": "appr-1", "decision": "approved"}
        )
        assert result["status"] == "delivered"

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_session_manager: MagicMock) -> None:
        handlers = Handlers(mock_session_manager)
        result = await handlers.handle_shutdown({})
        assert result["status"] == "shutdown"


class TestHandlersRegistration:
    def test_register_all(self, mock_session_manager: MagicMock) -> None:
        """All methods are registered with the dispatcher."""
        handlers = Handlers(mock_session_manager)
        dispatcher = MethodDispatcher()
        handlers.register_all(dispatcher)

        expected_methods = [
            "CreateSession",
            "StartTask",
            "CancelTask",
            "GetSessionState",
            "ApproveAction",
            "Shutdown",
        ]
        for method in expected_methods:
            assert method in dispatcher._handlers


class TestMethodDispatcher:
    @pytest.mark.asyncio
    async def test_dispatch_success(self, mock_session_manager: MagicMock) -> None:
        """Dispatcher routes to correct handler."""
        from agent_host.server.json_rpc import JsonRpcRequest

        handlers = Handlers(mock_session_manager)
        dispatcher = MethodDispatcher()
        handlers.register_all(dispatcher)

        request = JsonRpcRequest(method="GetSessionState", params={}, id=1)
        response = await dispatcher.dispatch(request)
        assert response.result is not None
        assert response.error is None

    @pytest.mark.asyncio
    async def test_dispatch_method_not_found(self) -> None:
        """Dispatcher returns error for unknown method."""
        from agent_host.server.json_rpc import JsonRpcRequest

        dispatcher = MethodDispatcher()
        request = JsonRpcRequest(method="UnknownMethod", params={}, id=1)
        response = await dispatcher.dispatch(request)
        assert response.error is not None
        assert response.error.code == -32601

    @pytest.mark.asyncio
    async def test_dispatch_agent_host_error(self) -> None:
        """AgentHostError maps to JSON-RPC error code."""
        from agent_host.exceptions import SessionNotFoundError
        from agent_host.server.json_rpc import JsonRpcRequest

        dispatcher = MethodDispatcher()

        async def failing_handler(params: dict) -> dict:
            raise SessionNotFoundError("No session")

        dispatcher.register("FailMethod", failing_handler)
        request = JsonRpcRequest(method="FailMethod", params={}, id=1)
        response = await dispatcher.dispatch(request)
        assert response.error is not None
        assert response.error.code == -32001  # SessionNotFoundError code

    @pytest.mark.asyncio
    async def test_dispatch_unexpected_error(self) -> None:
        """Unexpected errors become internal error."""
        from agent_host.server.json_rpc import JsonRpcRequest

        dispatcher = MethodDispatcher()

        async def crashing_handler(params: dict) -> dict:
            msg = "boom"
            raise RuntimeError(msg)

        dispatcher.register("CrashMethod", crashing_handler)
        request = JsonRpcRequest(method="CrashMethod", params={}, id=1)
        response = await dispatcher.dispatch(request)
        assert response.error is not None
        assert response.error.code == -32603
