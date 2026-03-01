"""HTTP client for the Session Service — session lifecycle management."""

from __future__ import annotations

import httpx
import structlog
from cowork_platform.session_cancel_request import SessionCancelRequest  # noqa: TC002
from cowork_platform.session_create_request import SessionCreateRequest  # noqa: TC002
from cowork_platform.session_create_response import SessionCreateResponse
from cowork_platform_sdk import create_http_client, raise_for_status
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
        return SessionCreateResponse.model_validate(response.json())

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
