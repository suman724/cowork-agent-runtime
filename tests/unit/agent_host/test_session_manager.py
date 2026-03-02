"""Tests for SessionManager — lifecycle coordination."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_host.config import AgentHostConfig
from agent_host.exceptions import (
    NoActiveTaskError,
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
            "approvalRules": [],
            "llmPolicy": {
                "allowedModels": ["gpt-4o"],
                "maxInputTokens": 8000,
                "maxOutputTokens": 4000,
                "maxSessionTokens": 100000,
            },
        },
    }


def _make_incompatible_response_dict() -> dict[str, object]:
    return {
        "sessionId": "sess-789",
        "workspaceId": "ws-abc",
        "compatibilityStatus": "incompatible",
    }


@pytest.fixture
def manager(tmp_path: object) -> SessionManager:
    config = _make_config(tmp_path)
    tool_router = MagicMock()
    tool_router.get_available_tools.return_value = [
        MagicMock(toolName="ReadFile"),
        MagicMock(toolName="WriteFile"),
    ]
    return SessionManager(config=config, tool_router=tool_router)


class TestCreateSession:
    async def test_create_session_success(self, manager: SessionManager) -> None:
        """Successfully create a session."""
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_session_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.create_session = AsyncMock(return_value=response)

        result = await manager.create_session({"tenantId": "t1", "userId": "u1"})

        assert result["sessionId"] == "sess-123"
        assert result["status"] == "ready"
        assert manager.session_context is not None
        assert manager.session_context.session_id == "sess-123"

    async def test_create_session_incompatible(self, manager: SessionManager) -> None:
        """Incompatible session returns status incompatible."""
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_incompatible_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.create_session = AsyncMock(return_value=response)

        result = await manager.create_session({"tenantId": "t1", "userId": "u1"})

        assert result["status"] == "incompatible"

    async def test_create_session_initializes_components(self, manager: SessionManager) -> None:
        """Components (LLM client, thread, etc.) should be initialized."""
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_session_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.create_session = AsyncMock(return_value=response)

        await manager.create_session({"tenantId": "t1", "userId": "u1"})

        assert manager._llm_client is not None
        assert manager._thread is not None
        assert manager._policy_enforcer is not None
        assert manager._token_budget is not None
        assert manager._approval_gate is not None
        assert manager._file_change_tracker is not None
        assert manager._event_emitter is not None


class TestResumeSession:
    async def test_resume_session(self, manager: SessionManager) -> None:
        """Should resume session and restore history."""
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_session_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.resume_session = AsyncMock(return_value=response)
        manager._workspace_client.get_session_history = AsyncMock(return_value=[])

        result = await manager.resume_session({"sessionId": "sess-123"})

        assert result["sessionId"] == "sess-123"
        assert result["status"] == "ready"

    async def test_resume_session_missing_id(self, manager: SessionManager) -> None:
        """Should raise SessionNotFoundError when sessionId is empty."""
        with pytest.raises(SessionNotFoundError):
            await manager.resume_session({})


class TestStartTask:
    async def test_start_task_no_session(self, manager: SessionManager) -> None:
        """Should raise SessionNotFoundError without an active session."""
        with pytest.raises(SessionNotFoundError):
            await manager.start_task({"taskId": "task-1", "prompt": "hi"})

    async def test_start_task_spawns_loop(self, manager: SessionManager) -> None:
        """Should spawn agent loop as background task."""
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_session_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.create_session = AsyncMock(return_value=response)

        await manager.create_session({"tenantId": "t1", "userId": "u1"})

        # Mock the _run_agent to avoid actually running the loop
        manager._run_agent = AsyncMock()  # type: ignore[method-assign]

        result = await manager.start_task(
            {
                "taskId": "task-1",
                "prompt": "Hello",
            }
        )

        assert result["taskId"] == "task-1"
        assert result["status"] == "running"
        assert manager._current_task is not None

    async def test_start_task_with_max_steps(self, manager: SessionManager) -> None:
        """Should parse maxSteps from taskOptions."""
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_session_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.create_session = AsyncMock(return_value=response)

        await manager.create_session({"tenantId": "t1", "userId": "u1"})
        manager._run_agent = AsyncMock()  # type: ignore[method-assign]

        await manager.start_task(
            {
                "taskId": "task-1",
                "prompt": "Hello",
                "taskOptions": {"maxSteps": 25},
            }
        )

        assert manager._current_max_steps == 25

    async def test_start_task_clamps_max_steps(self, manager: SessionManager) -> None:
        """maxSteps should be clamped to 1-200."""
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_session_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.create_session = AsyncMock(return_value=response)

        await manager.create_session({"tenantId": "t1", "userId": "u1"})
        manager._run_agent = AsyncMock()  # type: ignore[method-assign]

        await manager.start_task(
            {
                "taskId": "task-1",
                "prompt": "Hello",
                "taskOptions": {"maxSteps": 999},
            }
        )

        assert manager._current_max_steps == 200


class TestCancelTask:
    async def test_cancel_no_task(self, manager: SessionManager) -> None:
        """Should raise NoActiveTaskError."""
        with pytest.raises(NoActiveTaskError):
            await manager.cancel_task()

    async def test_cancel_running_task(self, manager: SessionManager) -> None:
        """Should cancel the running task and return status."""
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_session_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.create_session = AsyncMock(return_value=response)

        await manager.create_session({"tenantId": "t1", "userId": "u1"})

        # Create a long-running fake task
        async def long_task() -> None:
            await asyncio.sleep(10)

        manager._current_task = asyncio.create_task(long_task())
        manager._current_task_id = "task-1"

        result = await manager.cancel_task()
        assert result["status"] == "cancelled"
        assert result["taskId"] == "task-1"


class TestSessionState:
    async def test_no_session(self, manager: SessionManager) -> None:
        result = await manager.get_session_state()
        assert result["status"] == "no_session"

    async def test_with_session(self, manager: SessionManager) -> None:
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_session_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.create_session = AsyncMock(return_value=response)

        await manager.create_session({"tenantId": "t1", "userId": "u1"})

        result = await manager.get_session_state()
        assert result["sessionId"] == "sess-123"
        assert "tokenUsage" in result


class TestDeliverApproval:
    async def test_deliver_approval(self, manager: SessionManager) -> None:
        from agent_host.approval.approval_gate import ApprovalGate

        gate = ApprovalGate()
        manager._approval_gate = gate

        # Create a pending approval
        async def request():
            return await gate.request_approval("ap-1", timeout=5.0)

        task = asyncio.create_task(request())
        await asyncio.sleep(0.01)

        result = await manager.deliver_approval({"approvalId": "ap-1", "decision": "approved"})
        assert result["status"] == "delivered"
        assert result["decision"] == "approved"

        decision = await task
        assert decision == "approved"


class TestShutdown:
    async def test_shutdown_clean(self, manager: SessionManager) -> None:
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_session_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.create_session = AsyncMock(return_value=response)
        manager._session_client.close = AsyncMock()
        manager._workspace_client.close = AsyncMock()

        await manager.create_session({"tenantId": "t1", "userId": "u1"})

        result = await manager.shutdown()
        assert result["status"] == "shutdown"
        assert manager._session_context is None
        assert manager._llm_client is None

    async def test_shutdown_cancels_active_task(self, manager: SessionManager) -> None:
        from cowork_platform.session_create_response import SessionCreateResponse

        resp_dict = _make_session_response_dict()
        response = SessionCreateResponse.model_validate(resp_dict)
        manager._session_client.create_session = AsyncMock(return_value=response)
        manager._session_client.cancel_session = AsyncMock()
        manager._session_client.close = AsyncMock()
        manager._workspace_client.close = AsyncMock()

        await manager.create_session({"tenantId": "t1", "userId": "u1"})

        # Create a running task
        async def long_task() -> None:
            await asyncio.sleep(10)

        manager._current_task = asyncio.create_task(long_task())
        manager._current_task_id = "task-1"

        result = await manager.shutdown()
        assert result["status"] == "shutdown"
        manager._session_client.cancel_session.assert_awaited_once()
