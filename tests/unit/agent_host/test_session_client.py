"""Tests for SessionClient — HTTP client for Session Service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from cowork_platform.session_cancel_request import SessionCancelRequest
from cowork_platform.session_create_request import ClientInfo, SessionCreateRequest
from cowork_platform.session_create_response import SessionCreateResponse

from agent_host.session.session_client import SessionClient


def _make_session_request() -> SessionCreateRequest:
    return SessionCreateRequest(
        tenantId="tenant-1",
        userId="user-1",
        executionEnvironment="desktop",
        clientInfo=ClientInfo(
            desktopAppVersion="0.1.0",
            localAgentHostVersion="0.1.0",
            osFamily="macOS",
        ),
        supportedCapabilities=["File.Read", "LLM.Call"],
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
            "tenantId": "tenant-1",
            "userId": "user-1",
            "sessionId": "sess-123",
            "expiresAt": expires,
            "capabilities": [{"name": "File.Read"}, {"name": "LLM.Call"}],
            "llmPolicy": {
                "allowedModels": ["gpt-4o"],
                "maxInputTokens": 8000,
                "maxOutputTokens": 4000,
                "maxSessionTokens": 100000,
            },
            "approvalRules": [],
        },
    }


class TestSessionClient:
    @pytest.mark.asyncio
    async def test_create_session_success(self) -> None:
        """Successful session creation."""
        mock_response = httpx.Response(
            200,
            json=_make_session_response_dict(),
            request=httpx.Request("POST", "http://test/sessions"),
        )

        client = SessionClient("http://test")
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            response = await client.create_session(_make_session_request())

        assert isinstance(response, SessionCreateResponse)
        assert response.sessionId == "sess-123"
        assert response.workspaceId == "ws-456"
        await client.close()

    @pytest.mark.asyncio
    async def test_cancel_session_success(self) -> None:
        """Successful session cancellation."""
        mock_response = httpx.Response(
            200,
            json={},
            request=httpx.Request("POST", "http://test/sessions/sess-123/cancel"),
        )

        client = SessionClient("http://test")
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.cancel_session("sess-123", SessionCancelRequest(reason="shutdown"))

        mock_post.assert_called_once()
        await client.close()

    @pytest.mark.asyncio
    async def test_close_client(self) -> None:
        """Client can be closed."""
        client = SessionClient("http://test")
        with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_called_once()
