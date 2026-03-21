"""Tests for SQS consumer — session dispatch message polling and parsing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from agent_sdk.exceptions import SandboxStartupError

from agent_host.sandbox.sqs_consumer import (
    SqsSessionConfig,
    _parse_message,
    delete_message,
    poll_for_session,
)

QUEUE_URL = "http://localhost:4566/000000000000/dev-sandbox-requests"


def _make_sqs_message(
    body: dict | str | None = None,
    message_id: str = "msg-001",
    receipt_handle: str = "receipt-abc",
) -> dict:
    """Build a mock SQS message."""
    if body is None:
        body = {
            "sessionId": "sess-001",
            "registrationToken": "tok-abc",
            "sessionServiceUrl": "http://session:8000",
            "workspaceServiceUrl": "http://workspace:8002",
            "publishedAt": "2026-03-21T10:00:00Z",
        }
    body_str = json.dumps(body) if isinstance(body, dict) else body
    return {
        "MessageId": message_id,
        "ReceiptHandle": receipt_handle,
        "Body": body_str,
    }


# ---------------------------------------------------------------------------
# _parse_message
# ---------------------------------------------------------------------------


class TestParseMessage:
    def test_valid_message(self) -> None:
        msg = _make_sqs_message()
        config = _parse_message(msg, "receipt-abc")

        assert config.session_id == "sess-001"
        assert config.registration_token == "tok-abc"
        assert config.session_service_url == "http://session:8000"
        assert config.workspace_service_url == "http://workspace:8002"
        assert config.receipt_handle == "receipt-abc"

    def test_missing_session_id(self) -> None:
        msg = _make_sqs_message(body={"registrationToken": "tok"})
        with pytest.raises(SandboxStartupError, match="sessionId"):
            _parse_message(msg, "receipt")

    def test_missing_registration_token(self) -> None:
        msg = _make_sqs_message(body={"sessionId": "sess-001"})
        with pytest.raises(SandboxStartupError, match="registrationToken"):
            _parse_message(msg, "receipt")

    def test_invalid_json(self) -> None:
        msg = _make_sqs_message(body="not-json")
        with pytest.raises(SandboxStartupError, match="Invalid JSON"):
            _parse_message(msg, "receipt")

    def test_empty_body(self) -> None:
        msg = _make_sqs_message(body="")
        with pytest.raises(SandboxStartupError, match="Invalid JSON"):
            _parse_message(msg, "receipt")

    def test_optional_urls_default_to_empty(self) -> None:
        msg = _make_sqs_message(
            body={"sessionId": "s1", "registrationToken": "t1"}
        )
        config = _parse_message(msg, "receipt")
        assert config.session_service_url == ""
        assert config.workspace_service_url == ""


# ---------------------------------------------------------------------------
# poll_for_session
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPollForSession:
    async def test_returns_on_first_message(self) -> None:
        sqs = AsyncMock()
        sqs.receive_message.return_value = {
            "Messages": [_make_sqs_message()],
        }

        config = await poll_for_session(sqs, QUEUE_URL)

        assert config.session_id == "sess-001"
        assert config.registration_token == "tok-abc"
        sqs.receive_message.assert_called_once()

    async def test_skips_empty_polls(self) -> None:
        sqs = AsyncMock()
        # Two empty polls, then a message
        sqs.receive_message.side_effect = [
            {"Messages": []},
            {"Messages": []},
            {"Messages": [_make_sqs_message()]},
        ]

        config = await poll_for_session(sqs, QUEUE_URL)

        assert config.session_id == "sess-001"
        assert sqs.receive_message.call_count == 3

    @patch("agent_host.sandbox.sqs_consumer.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_receive_error(self, mock_sleep: AsyncMock) -> None:
        sqs = AsyncMock()
        # One error, then success
        sqs.receive_message.side_effect = [
            Exception("Connection timeout"),
            {"Messages": [_make_sqs_message()]},
        ]

        config = await poll_for_session(sqs, QUEUE_URL)

        assert config.session_id == "sess-001"
        assert sqs.receive_message.call_count == 2
        # Verify backoff was applied (2^1 = 2 seconds for first error)
        mock_sleep.assert_called_once_with(2)

    async def test_deletes_malformed_message_and_continues(self) -> None:
        sqs = AsyncMock()
        sqs.delete_message = AsyncMock()
        # First message is malformed, second is valid
        sqs.receive_message.side_effect = [
            {"Messages": [_make_sqs_message(body="bad-json")]},
            {"Messages": [_make_sqs_message()]},
        ]

        config = await poll_for_session(sqs, QUEUE_URL)

        assert config.session_id == "sess-001"
        # Malformed message should have been deleted
        sqs.delete_message.assert_called_once()
        assert sqs.receive_message.call_count == 2

    async def test_uses_correct_sqs_params(self) -> None:
        sqs = AsyncMock()
        sqs.receive_message.return_value = {
            "Messages": [_make_sqs_message()],
        }

        await poll_for_session(sqs, QUEUE_URL)

        call_kwargs = sqs.receive_message.call_args.kwargs
        assert call_kwargs["QueueUrl"] == QUEUE_URL
        assert call_kwargs["MaxNumberOfMessages"] == 1
        assert call_kwargs["WaitTimeSeconds"] == 20
        assert call_kwargs["VisibilityTimeout"] == 60


# ---------------------------------------------------------------------------
# delete_message
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteMessage:
    async def test_delete_success(self) -> None:
        sqs = AsyncMock()
        sqs.delete_message = AsyncMock()

        await delete_message(sqs, QUEUE_URL, "receipt-123")

        sqs.delete_message.assert_called_once_with(
            QueueUrl=QUEUE_URL,
            ReceiptHandle="receipt-123",
        )

    async def test_delete_failure_does_not_raise(self) -> None:
        sqs = AsyncMock()
        sqs.delete_message = AsyncMock(side_effect=Exception("access denied"))

        # Should not raise — best effort
        await delete_message(sqs, QUEUE_URL, "receipt-123")

        sqs.delete_message.assert_called_once()
