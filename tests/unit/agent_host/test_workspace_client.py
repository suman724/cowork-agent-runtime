"""Tests for WorkspaceClient — HTTP client for Workspace Service."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from cowork_platform.artifact import Artifact

from agent_host.session.workspace_client import WorkspaceClient


class TestWorkspaceClient:
    @pytest.mark.asyncio
    async def test_upload_artifact_success(self) -> None:
        """Successful artifact upload."""
        mock_response = httpx.Response(
            200,
            json={
                "artifactId": "art-123",
                "workspaceId": "ws-456",
                "sessionId": "sess-789",
                "artifactType": "tool_output",
                "createdAt": datetime.now(tz=UTC).isoformat(),
            },
            request=httpx.Request("POST", "http://test/workspaces/ws-456/artifacts"),
        )

        client = WorkspaceClient("http://test")
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            artifact = await client.upload_artifact(
                workspace_id="ws-456",
                session_id="sess-789",
                artifact_data=b"test data",
                artifact_type="tool_output",
                artifact_name="output.txt",
            )

        assert isinstance(artifact, Artifact)
        assert artifact.artifactId == "art-123"
        await client.close()

    @pytest.mark.asyncio
    async def test_upload_artifact_invalid_json(self) -> None:
        """Raises AgentHostError on malformed JSON response from artifact upload."""
        from agent_host.exceptions import AgentHostError

        mock_response = httpx.Response(
            200,
            content=b"not-json",
            request=httpx.Request("POST", "http://test/workspaces/ws-456/artifacts"),
        )

        client = WorkspaceClient("http://test")
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(AgentHostError, match="Invalid response"):
                await client.upload_artifact(
                    workspace_id="ws-456",
                    session_id="sess-789",
                    artifact_data=b"test data",
                )
        await client.close()

    @pytest.mark.asyncio
    async def test_upload_session_history_returns_none_on_failure(self) -> None:
        """Session history upload returns None on failure (best-effort)."""
        client = WorkspaceClient("http://test")
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TransportError("Connection refused")
            result = await client.upload_session_history(
                workspace_id="ws-456",
                session_id="sess-789",
                messages=[],
            )

        assert result is None
        await client.close()

    @pytest.mark.asyncio
    async def test_close_client(self) -> None:
        """Client can be closed."""
        client = WorkspaceClient("http://test")
        with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_called_once()
