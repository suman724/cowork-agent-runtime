"""Tests for sandbox shutdown ordering in run_http().

Verifies that workspace upload happens BEFORE session_manager.shutdown()
on SIGTERM/SIGINT, preventing data loss from uploading after session
cancellation.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_host.config import AgentHostConfig
from agent_host.sandbox.startup import RegistrationResult


def _make_config(**overrides: Any) -> AgentHostConfig:
    defaults = {
        "llm_gateway_endpoint": "http://llm:8080",
        "llm_gateway_auth_token": "test-token",
        "session_service_url": "http://session:8000",
        "workspace_service_url": "http://workspace:8000",
        "session_id": "sess-123",
        "registration_token": "reg-tok",
        "sandbox_local_mode": True,
    }
    defaults.update(overrides)
    return AgentHostConfig(**defaults)


def _make_args(**overrides: Any) -> argparse.Namespace:
    defaults = {
        "transport": "http",
        "host": "0.0.0.0",  # noqa: S104
        "port": 8080,
        "workspace_dir": "/tmp/workspace",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


_REG_RESULT = RegistrationResult(
    session_id="sess-123",
    workspace_id="ws-456",
    workspace_service_url="http://workspace:8000",
    policy_bundle={"capabilities": [], "llmPolicy": None},
)


@pytest.mark.asyncio
async def test_shutdown_uploads_before_session_shutdown() -> None:
    """Workspace upload must complete BEFORE session_manager.shutdown()."""
    call_order: list[str] = []

    async def mock_upload(*_a: Any, **_kw: Any) -> None:
        call_order.append("upload_workspace")

    async def mock_shutdown() -> None:
        call_order.append("session_manager.shutdown")

    mock_transport = MagicMock()
    mock_transport.start = AsyncMock()
    mock_transport.set_ready = MagicMock()
    mock_transport.set_dispatcher = MagicMock()
    mock_transport.shutdown = AsyncMock()

    mock_sm = MagicMock()
    mock_sm.init_from_registration = AsyncMock()
    mock_sm.shutdown = AsyncMock(side_effect=mock_shutdown)

    mock_client = AsyncMock()
    mock_client.close = AsyncMock()

    config = _make_config()
    args = _make_args()

    with (
        patch(
            "agent_host.events.event_buffer.EventBuffer",
            return_value=MagicMock(),
        ),
        patch(
            "agent_host.transport.http_transport.HttpTransport",
            return_value=mock_transport,
        ),
        patch("agent_host.main.ToolRouter"),
        patch("agent_host.main.SessionManager", return_value=mock_sm),
        patch("agent_host.main.MethodDispatcher"),
        patch("agent_host.main.Handlers"),
        patch(
            "agent_host.sandbox.startup.run_sandbox_startup",
            new_callable=AsyncMock,
            return_value=_REG_RESULT,
        ),
        patch(
            "agent_host.sandbox.workspace_sync.download_workspace",
            new_callable=AsyncMock,
        ),
        patch(
            "agent_host.sandbox.workspace_sync.upload_workspace",
            new_callable=AsyncMock,
            side_effect=mock_upload,
        ),
        patch(
            "agent_host.session.session_client.SessionClient",
            return_value=mock_client,
        ),
    ):
        from agent_host.main import run_http

        task = asyncio.create_task(run_http(config, args))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "upload_workspace" in call_order
    assert "session_manager.shutdown" in call_order
    idx_upload = call_order.index("upload_workspace")
    idx_shutdown = call_order.index("session_manager.shutdown")
    assert idx_upload < idx_shutdown


@pytest.mark.asyncio
async def test_shutdown_upload_failure_does_not_block() -> None:
    """If workspace upload fails, session shutdown still runs."""
    call_order: list[str] = []

    async def mock_upload_fail(*_a: Any, **_kw: Any) -> None:
        call_order.append("upload_workspace")
        msg = "S3 error"
        raise RuntimeError(msg)

    async def mock_shutdown() -> None:
        call_order.append("session_manager.shutdown")

    mock_transport = MagicMock()
    mock_transport.start = AsyncMock()
    mock_transport.set_ready = MagicMock()
    mock_transport.set_dispatcher = MagicMock()
    mock_transport.shutdown = AsyncMock()

    mock_sm = MagicMock()
    mock_sm.init_from_registration = AsyncMock()
    mock_sm.shutdown = AsyncMock(side_effect=mock_shutdown)

    mock_client = AsyncMock()
    mock_client.close = AsyncMock()

    config = _make_config()
    args = _make_args()

    with (
        patch(
            "agent_host.events.event_buffer.EventBuffer",
            return_value=MagicMock(),
        ),
        patch(
            "agent_host.transport.http_transport.HttpTransport",
            return_value=mock_transport,
        ),
        patch("agent_host.main.ToolRouter"),
        patch("agent_host.main.SessionManager", return_value=mock_sm),
        patch("agent_host.main.MethodDispatcher"),
        patch("agent_host.main.Handlers"),
        patch(
            "agent_host.sandbox.startup.run_sandbox_startup",
            new_callable=AsyncMock,
            return_value=_REG_RESULT,
        ),
        patch(
            "agent_host.sandbox.workspace_sync.download_workspace",
            new_callable=AsyncMock,
        ),
        patch(
            "agent_host.sandbox.workspace_sync.upload_workspace",
            new_callable=AsyncMock,
            side_effect=mock_upload_fail,
        ),
        patch(
            "agent_host.session.session_client.SessionClient",
            return_value=mock_client,
        ),
    ):
        from agent_host.main import run_http

        task = asyncio.create_task(run_http(config, args))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # Upload failed but shutdown still ran
    assert "upload_workspace" in call_order
    assert "session_manager.shutdown" in call_order
