"""HTTP client for the Workspace Service — artifact and history uploads."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import structlog
from cowork_platform.artifact import Artifact
from cowork_platform.artifact_upload_request import ArtifactUploadRequest
from cowork_platform_sdk import create_http_client, raise_for_status
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent_host.exceptions import AgentHostError

if TYPE_CHECKING:
    from cowork_platform.conversation_message import ConversationMessage

logger = structlog.get_logger()


class WorkspaceClient:
    """HTTP client for the Workspace Service.

    Handles artifact uploads and session history persistence.
    All operations are best-effort — failures are logged but don't
    break the agent loop.
    """

    def __init__(self, base_url: str) -> None:
        self._client: httpx.AsyncClient = create_http_client(base_url)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    async def upload_artifact(
        self,
        workspace_id: str,
        session_id: str,
        artifact_data: bytes,
        artifact_type: str = "tool_output",
        artifact_name: str | None = None,
        content_type: str = "text/plain",
        task_id: str | None = None,
        step_id: str | None = None,
    ) -> Artifact:
        """Upload an artifact to the Workspace Service.

        Retries on transient network/timeout errors (max 3 attempts).
        """
        encoded = base64.b64encode(artifact_data).decode("ascii")
        request = ArtifactUploadRequest(
            sessionId=session_id,
            taskId=task_id,
            stepId=step_id,
            artifactType=artifact_type,
            artifactName=artifact_name,
            contentType=content_type,
            contentBase64=encoded,
        )
        response = await self._client.post(
            f"/workspaces/{workspace_id}/artifacts",
            content=request.model_dump_json(),
        )
        await raise_for_status(response)
        try:
            return Artifact.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise AgentHostError(
                f"Invalid response from Workspace Service: {exc}",
            ) from exc

    async def upload_session_history(
        self,
        workspace_id: str,
        session_id: str,
        messages: list[ConversationMessage],
        task_id: str | None = None,
    ) -> dict[str, str] | None:
        """Upload session history as an artifact. Best-effort — returns None on failure."""
        try:
            # Convert messages to dicts because ArtifactUploadRequest has its
            # own generated ConversationMessage class (separate from
            # cowork_platform.conversation_message.ConversationMessage).
            message_dicts = [m.model_dump(mode="json") for m in messages]
            request = ArtifactUploadRequest(
                sessionId=session_id,
                taskId=task_id,
                artifactType="session_history",
                artifactName=f"history-{session_id}",
                contentType="application/json",
                messages=message_dicts,
                snapshotAt=datetime.now(tz=UTC),
            )
            response = await self._client.post(
                f"/workspaces/{workspace_id}/artifacts",
                content=request.model_dump_json(),
            )
            await raise_for_status(response)
            # Workspace service returns {"artifactId": "...", "artifactUri": "..."}
            result: dict[str, str] = response.json()
            return result
        except Exception:
            logger.warning(
                "workspace_client.upload_session_history_failed",
                workspace_id=workspace_id,
                session_id=session_id,
                exc_info=True,
            )
            return None
