"""SQS consumer for sandbox session dispatch.

Polls an SQS queue for session requests. When a message is received,
extracts the session config (sessionId, registrationToken, service URLs)
and returns it for the sandbox startup flow to use.

This module handles only the SQS polling and message parsing. The actual
sandbox startup (self-registration, workspace sync, etc.) remains in
startup.py. See docs/design/sqs-sandbox-dispatch.md for the full design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog
from agent_sdk.exceptions import SandboxStartupError

logger = structlog.get_logger()

# SQS long-poll wait time (max 20 seconds per AWS docs)
_SQS_WAIT_TIME_SECONDS = 20
# Maximum messages per receive (we only process one at a time)
_SQS_MAX_MESSAGES = 1
# Visibility timeout — message invisible to other consumers during processing
# Set higher than time needed to delete the message (a few seconds)
_SQS_VISIBILITY_TIMEOUT = 60


@dataclass(frozen=True)
class SqsSessionConfig:
    """Session configuration extracted from an SQS message."""

    session_id: str
    registration_token: str
    session_service_url: str
    workspace_service_url: str
    receipt_handle: str  # Needed for message deletion


async def poll_for_session(
    sqs_client: Any,
    queue_url: str,
) -> SqsSessionConfig:
    """Poll SQS until a session request message is received.

    Long-polls the queue with 20-second wait time. Returns when a valid
    message is received. Retries indefinitely on empty polls (normal
    behavior when no sessions are waiting).

    The caller is responsible for deleting the message after successful
    processing (via delete_message).

    Args:
        sqs_client: An aioboto3 SQS client.
        queue_url: The SQS queue URL to poll.

    Returns:
        SqsSessionConfig with session details from the message.

    Raises:
        SandboxStartupError: If the message body is malformed.
    """
    logger.info("sqs_polling_started", queue_url=queue_url)

    while True:
        try:
            response = await sqs_client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=_SQS_MAX_MESSAGES,
                WaitTimeSeconds=_SQS_WAIT_TIME_SECONDS,
                VisibilityTimeout=_SQS_VISIBILITY_TIMEOUT,
            )
        except Exception as exc:
            # Log and retry — transient SQS errors should not crash the worker
            logger.warning("sqs_receive_error", error=str(exc))
            continue

        messages = response.get("Messages", [])
        if not messages:
            # Empty poll — normal when no sessions are waiting. Continue polling.
            continue

        message = messages[0]
        receipt_handle = message.get("ReceiptHandle", "")

        try:
            config = _parse_message(message, receipt_handle)
        except SandboxStartupError:
            # Malformed message — delete it so it doesn't block the queue.
            # After maxReceiveCount failures, SQS moves it to DLQ automatically.
            logger.error(
                "sqs_message_malformed",
                message_id=message.get("MessageId"),
                receipt_handle=receipt_handle,
            )
            await _safe_delete(sqs_client, queue_url, receipt_handle)
            continue

        logger.info(
            "sqs_session_received",
            session_id=config.session_id,
            message_id=message.get("MessageId"),
        )
        return config


async def delete_message(
    sqs_client: Any,
    queue_url: str,
    receipt_handle: str,
) -> None:
    """Delete a processed message from the SQS queue.

    Called immediately after receiving a valid message, before
    self-registration. If the worker crashes between receive and delete,
    the message becomes visible again after the visibility timeout — but
    the provisioning timeout (180s) in the lifecycle manager will catch
    the orphaned session.
    """
    try:
        await sqs_client.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle,
        )
        logger.debug("sqs_message_deleted", receipt_handle=receipt_handle[:20])
    except Exception as exc:
        # Non-fatal — message will become visible again after visibility timeout.
        # The session registration will succeed (idempotent), and the duplicate
        # message will fail registration (session already in SANDBOX_READY).
        logger.warning("sqs_delete_failed", error=str(exc))


def _parse_message(message: dict[str, Any], receipt_handle: str) -> SqsSessionConfig:
    """Parse and validate an SQS message body into SqsSessionConfig.

    Raises SandboxStartupError if required fields are missing.
    """
    body_str = message.get("Body", "")
    try:
        body = json.loads(body_str)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SandboxStartupError(f"Invalid JSON in SQS message: {exc}") from exc

    session_id = body.get("sessionId")
    if not session_id:
        raise SandboxStartupError("SQS message missing 'sessionId'")

    registration_token = body.get("registrationToken")
    if not registration_token:
        raise SandboxStartupError("SQS message missing 'registrationToken'")

    session_service_url = body.get("sessionServiceUrl", "")
    workspace_service_url = body.get("workspaceServiceUrl", "")

    return SqsSessionConfig(
        session_id=session_id,
        registration_token=registration_token,
        session_service_url=session_service_url,
        workspace_service_url=workspace_service_url,
        receipt_handle=receipt_handle,
    )


async def _safe_delete(sqs_client: Any, queue_url: str, receipt_handle: str) -> None:
    """Best-effort delete — don't let delete failure crash the poll loop."""
    try:
        await sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
    except Exception as exc:
        logger.warning("sqs_safe_delete_failed", error=str(exc))
