"""Tests for sandbox self-registration startup flow."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from agent_sdk.exceptions import SandboxStartupError

from agent_host.config import AgentHostConfig
from agent_host.sandbox.startup import (
    EcsMetadata,
    RegistrationResult,
    _fetch_ecs_metadata,
    run_sandbox_startup,
)


def _mock_response(status_code: int, **kwargs: Any) -> httpx.Response:
    """Create an httpx.Response with a fake request set (needed for raise_for_status)."""
    resp = httpx.Response(status_code, **kwargs)
    resp._request = httpx.Request("GET", "http://test")
    return resp


def _make_config(**overrides: Any) -> AgentHostConfig:
    """Create a test config with sandbox fields."""
    defaults = {
        "llm_gateway_endpoint": "http://llm:8080",
        "llm_gateway_auth_token": "test-token",
        "session_service_url": "http://session:8000",
        "workspace_service_url": "http://workspace:8000",
        "session_id": "sess-123",
        "registration_token": "reg-tok-abc",
        "sandbox_local_mode": False,
    }
    defaults.update(overrides)
    return AgentHostConfig(**defaults)


def _mock_session_client(
    return_value: dict[str, Any] | None = None,
    side_effect: Exception | None = None,
) -> AsyncMock:
    """Create a mock SessionClient for registration tests."""
    client = AsyncMock()
    if side_effect:
        client.register_sandbox = AsyncMock(side_effect=side_effect)
    else:
        client.register_sandbox = AsyncMock(
            return_value=return_value
            or {
                "sessionId": "sess-123",
                "workspaceId": "ws-456",
                "workspaceServiceUrl": "http://workspace:8000",
                "policyBundle": {"capabilities": [], "llmPolicy": None},
            }
        )
    return client


# -- _fetch_ecs_metadata tests --


@pytest.mark.asyncio
async def test_fetch_ecs_metadata_missing_env() -> None:
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(SandboxStartupError, match="ECS_CONTAINER_METADATA_URI_V4"),
    ):
        await _fetch_ecs_metadata()


@pytest.mark.asyncio
async def test_fetch_ecs_metadata_success() -> None:
    container_resp = _mock_response(200, json={"Networks": [{"IPv4Addresses": ["10.0.1.42"]}]})
    task_resp = _mock_response(
        200, json={"TaskARN": "arn:aws:ecs:us-east-1:123:task/cluster/abc123"}
    )

    with (
        patch.dict("os.environ", {"ECS_CONTAINER_METADATA_URI_V4": "http://169.254.170.2/v4"}),
        patch("agent_host.sandbox.startup.httpx.AsyncClient") as mock_client_cls,
    ):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[container_resp, task_resp])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await _fetch_ecs_metadata()
        assert isinstance(result, EcsMetadata)
        assert result.container_ip == "10.0.1.42"
        assert result.task_arn == "arn:aws:ecs:us-east-1:123:task/cluster/abc123"
        assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_fetch_ecs_metadata_no_networks() -> None:
    container_resp = _mock_response(200, json={"Networks": []})
    task_resp = _mock_response(200, json={"TaskARN": "arn:task"})

    with (
        patch.dict("os.environ", {"ECS_CONTAINER_METADATA_URI_V4": "http://169.254.170.2/v4"}),
        patch("agent_host.sandbox.startup.httpx.AsyncClient") as mock_client_cls,
    ):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[container_resp, task_resp])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        with pytest.raises(SandboxStartupError, match="No networks"):
            await _fetch_ecs_metadata()


@pytest.mark.asyncio
async def test_fetch_ecs_metadata_missing_task_arn() -> None:
    container_resp = _mock_response(200, json={"Networks": [{"IPv4Addresses": ["10.0.1.42"]}]})
    task_resp = _mock_response(200, json={"TaskARN": ""})

    with (
        patch.dict("os.environ", {"ECS_CONTAINER_METADATA_URI_V4": "http://169.254.170.2/v4"}),
        patch("agent_host.sandbox.startup.httpx.AsyncClient") as mock_client_cls,
    ):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[container_resp, task_resp])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        with pytest.raises(SandboxStartupError, match="TaskARN not found"):
            await _fetch_ecs_metadata()


# -- run_sandbox_startup tests --


@pytest.mark.asyncio
async def test_startup_missing_session_id() -> None:
    config = _make_config(session_id="")
    mock_client = _mock_session_client()
    with pytest.raises(SandboxStartupError, match="SESSION_ID"):
        await run_sandbox_startup(config, mock_client, port=8080)


@pytest.mark.asyncio
async def test_startup_local_mode_success() -> None:
    """Local mode: skip ECS metadata, use localhost, register with Session Service."""
    config = _make_config(sandbox_local_mode=True)
    mock_client = _mock_session_client()

    result = await run_sandbox_startup(config, mock_client, port=8080)

    assert isinstance(result, RegistrationResult)
    assert result.session_id == "sess-123"
    assert result.workspace_id == "ws-456"
    assert result.policy_bundle == {"capabilities": [], "llmPolicy": None}

    # Verify registration was called with correct args
    mock_client.register_sandbox.assert_called_once()
    call_args = mock_client.register_sandbox.call_args
    assert call_args.args == ("sess-123",)
    assert call_args.kwargs["sandbox_endpoint"] == "http://127.0.0.1:8080"
    assert "task_arn" not in call_args.kwargs  # task_arn removed in SQS dispatch
    assert call_args.kwargs["registration_token"] == "reg-tok-abc"  # noqa: S105


@pytest.mark.asyncio
async def test_startup_registration_error() -> None:
    config = _make_config(sandbox_local_mode=True)
    mock_client = _mock_session_client(
        side_effect=httpx.ConnectError("refused"),
    )

    with pytest.raises(SandboxStartupError, match="Registration failed"):
        await run_sandbox_startup(config, mock_client, port=8080)


@pytest.mark.asyncio
async def test_startup_missing_policy_bundle() -> None:
    config = _make_config(sandbox_local_mode=True)
    mock_client = _mock_session_client(
        return_value={"sessionId": "sess-123", "workspaceId": "ws-456"},
    )

    with pytest.raises(SandboxStartupError, match="missing policyBundle"):
        await run_sandbox_startup(config, mock_client, port=8080)


@pytest.mark.asyncio
async def test_startup_no_registration_token() -> None:
    """Registration token is optional — should be passed as None if empty."""
    config = _make_config(sandbox_local_mode=True, registration_token="")
    mock_client = _mock_session_client()

    result = await run_sandbox_startup(config, mock_client, port=8080)
    assert result.session_id == "sess-123"

    # Verify registration_token was passed as None
    mock_client.register_sandbox.assert_called_once()
    call_args = mock_client.register_sandbox.call_args
    assert call_args.args == ("sess-123",)
    assert call_args.kwargs["sandbox_endpoint"] == "http://127.0.0.1:8080"
    assert "task_arn" not in call_args.kwargs  # task_arn removed in SQS dispatch
    assert call_args.kwargs["registration_token"] is None
