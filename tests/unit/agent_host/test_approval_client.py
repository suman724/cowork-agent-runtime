"""Tests for ApprovalClient — fire-and-forget decision persistence."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agent_host.approval.approval_client import ApprovalClient


def _make_success_response() -> httpx.Response:
    return httpx.Response(201, json={"approvalId": "apr-1", "decision": "approved"})


def _make_error_response(status: int = 500) -> httpx.Response:
    return httpx.Response(
        status,
        json={"code": "INTERNAL_ERROR", "message": "fail", "retryable": status >= 500},
    )


@pytest.mark.unit
class TestApprovalClientPersist:
    async def test_persist_decision_posts_to_approval_service(self) -> None:
        client = ApprovalClient("http://test:8003")
        mock_post = AsyncMock(return_value=_make_success_response())
        client._client.post = mock_post  # type: ignore[method-assign]

        await client._persist_decision(
            approval_id="apr-1",
            session_id="sess-1",
            task_id="task-1",
            user_id="u1",
            tenant_id="t1",
            workspace_id="ws-1",
            decision="approved",
            action_summary="WriteFile (File.Write)",
            risk_level="medium",
            client_timestamp=datetime(2026, 3, 8, 12, 0, tzinfo=UTC),
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["approvalId"] == "apr-1"
        assert payload["decision"] == "approved"
        assert payload["sessionId"] == "sess-1"
        assert payload["tenantId"] == "t1"
        assert payload["actionSummary"] == "WriteFile (File.Write)"
        assert payload["riskLevel"] == "medium"

    async def test_persist_decision_omits_optional_fields_when_none(self) -> None:
        client = ApprovalClient("http://test:8003")
        mock_post = AsyncMock(return_value=_make_success_response())
        client._client.post = mock_post  # type: ignore[method-assign]

        await client._persist_decision(
            approval_id="apr-1",
            session_id="sess-1",
            task_id="task-1",
            user_id="u1",
            tenant_id="t1",
            workspace_id="ws-1",
            decision="denied",
            action_summary="DeleteFile (File.Delete)",
            risk_level=None,
            client_timestamp=datetime(2026, 3, 8, 12, 0, tzinfo=UTC),
        )

        payload = mock_post.call_args.kwargs["json"]
        assert "riskLevel" not in payload
        assert "stepId" not in payload
        assert "reason" not in payload

    async def test_persist_decision_includes_optional_fields(self) -> None:
        client = ApprovalClient("http://test:8003")
        mock_post = AsyncMock(return_value=_make_success_response())
        client._client.post = mock_post  # type: ignore[method-assign]

        await client._persist_decision(
            approval_id="apr-1",
            session_id="sess-1",
            task_id="task-1",
            step_id="step-1",
            user_id="u1",
            tenant_id="t1",
            workspace_id="ws-1",
            decision="denied",
            reason="Too risky",
            action_summary="RunCommand (Shell.Exec)",
            risk_level="high",
            client_timestamp=datetime(2026, 3, 8, 12, 0, tzinfo=UTC),
        )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["stepId"] == "step-1"
        assert payload["reason"] == "Too risky"
        assert payload["riskLevel"] == "high"


@pytest.mark.unit
class TestApprovalClientFireAndForget:
    async def test_background_persist_does_not_raise_on_failure(self) -> None:
        """Fire-and-forget must never propagate errors."""
        client = ApprovalClient("http://test:8003")
        client._client.post = AsyncMock(  # type: ignore[method-assign]
            side_effect=httpx.ConnectError("connection refused")
        )

        # This should not raise
        client.persist_decision_background(
            approval_id="apr-1",
            session_id="sess-1",
            task_id="task-1",
            user_id="u1",
            tenant_id="t1",
            workspace_id="ws-1",
            decision="approved",
            action_summary="WriteFile (File.Write)",
            client_timestamp=datetime(2026, 3, 8, 12, 0, tzinfo=UTC),
        )

        # Let the background task run
        await asyncio.sleep(0.1)

    async def test_background_persist_calls_persist(self) -> None:
        client = ApprovalClient("http://test:8003")
        mock_post = AsyncMock(return_value=_make_success_response())
        client._client.post = mock_post  # type: ignore[method-assign]

        client.persist_decision_background(
            approval_id="apr-1",
            session_id="sess-1",
            task_id="task-1",
            user_id="u1",
            tenant_id="t1",
            workspace_id="ws-1",
            decision="approved",
            action_summary="WriteFile (File.Write)",
            client_timestamp=datetime(2026, 3, 8, 12, 0, tzinfo=UTC),
        )

        await asyncio.sleep(0.1)
        mock_post.assert_called_once()


@pytest.mark.unit
class TestApprovalClientRetry:
    async def test_retries_on_transport_error(self) -> None:
        client = ApprovalClient("http://test:8003")
        # Fail twice with transport error, succeed on third attempt
        client._client.post = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                httpx.ConnectError("refused"),
                httpx.ConnectError("refused"),
                _make_success_response(),
            ]
        )

        # Patch tenacity to not actually wait
        with patch("agent_host.approval.approval_client.wait_exponential", return_value=0):
            await client._persist_decision(
                approval_id="apr-1",
                session_id="sess-1",
                task_id="task-1",
                user_id="u1",
                tenant_id="t1",
                workspace_id="ws-1",
                decision="approved",
                action_summary="test",
                client_timestamp=datetime(2026, 3, 8, 12, 0, tzinfo=UTC),
            )

        assert client._client.post.call_count == 3  # type: ignore[union-attr]

    async def test_gives_up_after_max_retries(self) -> None:
        client = ApprovalClient("http://test:8003")
        client._client.post = AsyncMock(  # type: ignore[method-assign]
            side_effect=httpx.ConnectError("refused")
        )

        with pytest.raises(httpx.ConnectError):
            await client._persist_decision(
                approval_id="apr-1",
                session_id="sess-1",
                task_id="task-1",
                user_id="u1",
                tenant_id="t1",
                workspace_id="ws-1",
                decision="approved",
                action_summary="test",
                client_timestamp=datetime(2026, 3, 8, 12, 0, tzinfo=UTC),
            )


@pytest.mark.unit
class TestApprovalClientClose:
    async def test_close(self) -> None:
        client = ApprovalClient("http://test:8003")
        client._client = AsyncMock()  # type: ignore[assignment]
        await client.close()
        client._client.aclose.assert_called_once()
