"""HTTP client for the Approval Service — fire-and-forget decision persistence."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import structlog
from cowork_platform_sdk import CoworkAPIError, create_http_client
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from datetime import datetime

logger = structlog.get_logger()


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient transport errors and retryable API errors (5xx)."""
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    return isinstance(exc, CoworkAPIError) and exc.retryable


class ApprovalClient:
    """HTTP client for persisting approval decisions to the Approval Service.

    Persistence is fire-and-forget with retry — tool execution is never
    blocked on persistence succeeding. The decision is already made locally.
    """

    def __init__(self, base_url: str) -> None:
        self._client: httpx.AsyncClient = create_http_client(base_url)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def persist_decision_background(
        self,
        *,
        approval_id: str,
        session_id: str,
        task_id: str,
        step_id: str | None = None,
        user_id: str,
        tenant_id: str,
        workspace_id: str,
        decision: str,
        reason: str | None = None,
        action_summary: str,
        risk_level: str | None = None,
        client_timestamp: datetime,
    ) -> None:
        """Fire-and-forget: schedule decision persistence as a background task.

        Failures are logged but never propagate — tool execution must not block.
        """

        async def _safe_persist() -> None:
            try:
                await self._persist_decision(
                    approval_id=approval_id,
                    session_id=session_id,
                    task_id=task_id,
                    step_id=step_id,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    decision=decision,
                    reason=reason,
                    action_summary=action_summary,
                    risk_level=risk_level,
                    client_timestamp=client_timestamp,
                )
            except Exception:
                logger.warning(
                    "approval_persist_failed",
                    approval_id=approval_id,
                    exc_info=True,
                )

        asyncio.create_task(_safe_persist())  # noqa: RUF006

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async def _persist_decision(
        self,
        *,
        approval_id: str,
        session_id: str,
        task_id: str,
        step_id: str | None = None,
        user_id: str,
        tenant_id: str,
        workspace_id: str,
        decision: str,
        reason: str | None = None,
        action_summary: str,
        risk_level: str | None = None,
        client_timestamp: datetime,
    ) -> None:
        """POST the approval decision to the Approval Service with retry."""
        from cowork_platform_sdk import raise_for_status

        payload: dict[str, object] = {
            "approvalId": approval_id,
            "sessionId": session_id,
            "taskId": task_id,
            "userId": user_id,
            "tenantId": tenant_id,
            "workspaceId": workspace_id,
            "decision": decision,
            "actionSummary": action_summary,
            "clientTimestamp": client_timestamp.isoformat(),
        }
        if step_id:
            payload["stepId"] = step_id
        if reason:
            payload["reason"] = reason
        if risk_level:
            payload["riskLevel"] = risk_level

        response = await self._client.post("/approvals", json=payload)
        await raise_for_status(response)

        logger.info(
            "approval_persisted",
            approval_id=approval_id,
            decision=decision,
        )
