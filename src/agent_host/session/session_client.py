"""HTTP client for the Session Service — session lifecycle management."""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from cowork_platform.session_cancel_request import SessionCancelRequest  # noqa: TC002
from cowork_platform.session_create_request import SessionCreateRequest  # noqa: TC002
from cowork_platform.session_create_response import SessionCreateResponse
from cowork_platform_sdk import create_http_client, raise_for_status
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent_host.exceptions import AgentHostError

logger = structlog.get_logger()


class SessionClient:
    """HTTP client for the Session Service.

    Handles session creation, resume, and cancellation with
    automatic retry on transient failures.
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
    async def create_session(self, request: SessionCreateRequest) -> SessionCreateResponse:
        """Create a new session via POST /sessions.

        Retries on transient network/timeout errors (max 3 attempts).
        """
        logger.info("session_client.create_session", tenant_id=request.tenantId)
        response = await self._client.post(
            "/sessions",
            content=request.model_dump_json(),
        )
        await raise_for_status(response)
        try:
            return SessionCreateResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise AgentHostError(
                f"Invalid response from Session Service: {exc}",
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    async def resume_session(self, session_id: str) -> SessionCreateResponse:
        """Resume an existing session via POST /sessions/{id}/resume.

        Returns the same response shape as create_session (sessionId, workspaceId, policyBundle).
        """
        logger.info("session_client.resume_session", session_id=session_id)
        response = await self._client.post(f"/sessions/{session_id}/resume")
        await raise_for_status(response)
        try:
            return SessionCreateResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise AgentHostError(
                f"Invalid response from Session Service: {exc}",
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    async def update_session_name(
        self, session_id: str, name: str, auto_named: bool = True
    ) -> None:
        """Update session name via PATCH /sessions/{id}/name."""
        logger.info("session_client.update_session_name", session_id=session_id, name=name)
        response = await self._client.patch(
            f"/sessions/{session_id}/name",
            json={"name": name, "autoNamed": auto_named},
        )
        await raise_for_status(response)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    async def create_task(
        self, session_id: str, task_id: str, prompt: str, max_steps: int = 50
    ) -> None:
        """Report task creation via POST /sessions/{id}/tasks."""
        logger.info("session_client.create_task", session_id=session_id, task_id=task_id)
        response = await self._client.post(
            f"/sessions/{session_id}/tasks",
            json={"taskId": task_id, "prompt": prompt, "maxSteps": max_steps},
        )
        await raise_for_status(response)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    async def complete_task(
        self,
        session_id: str,
        task_id: str,
        status: str,
        step_count: int = 0,
        completion_reason: str | None = None,
    ) -> None:
        """Report task completion via POST /sessions/{id}/tasks/{task_id}/complete."""
        logger.info(
            "session_client.complete_task",
            session_id=session_id,
            task_id=task_id,
            status=status,
        )
        body: dict[str, Any] = {"status": status, "stepCount": step_count}
        if completion_reason:
            body["completionReason"] = completion_reason
        response = await self._client.post(
            f"/sessions/{session_id}/tasks/{task_id}/complete",
            json=body,
        )
        await raise_for_status(response)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    async def cancel_session(self, session_id: str, request: SessionCancelRequest) -> None:
        """Cancel a session via POST /sessions/{id}/cancel.

        Retries on transient network/timeout errors (max 3 attempts).
        """
        logger.info("session_client.cancel_session", session_id=session_id)
        response = await self._client.post(
            f"/sessions/{session_id}/cancel",
            content=request.model_dump_json(),
        )
        await raise_for_status(response)
