"""Tests for SessionManager — _run_agent exception handling with enriched error events."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_host.events.event_emitter import EventEmitter
from agent_host.exceptions import CheckpointError, PolicyExpiredError
from agent_host.models import SessionContext
from agent_host.session.session_manager import SessionManager


def _make_session_manager() -> SessionManager:
    """Build a minimal SessionManager with mocked dependencies."""
    config = MagicMock()
    config.session_service_url = "http://localhost:8001"
    config.workspace_service_url = "http://localhost:8002"
    config.checkpoint_dir = "/tmp/ckpt"
    config.llm_gateway_endpoint = "http://localhost:8080"
    config.llm_gateway_auth_token = "test-token"  # noqa: S105
    config.llm_model = "openai/gpt-4o"
    config.llm_max_retries = 3
    config.llm_retry_base_delay = 0.1
    config.llm_retry_max_delay = 1.0
    config.default_max_steps = 50
    config.recency_window = 10
    config.max_context_tokens = 128000
    config.approval_timeout_seconds = 300
    config.log_dir = "/tmp/logs"

    tool_router = MagicMock()

    sm = SessionManager(config, tool_router)

    # Set up session context
    sm._session_context = SessionContext(
        session_id="sess-1",
        workspace_id="ws-1",
        tenant_id="t-1",
        user_id="u-1",
    )

    # Mock event emitter
    sm._event_emitter = MagicMock(spec=EventEmitter)

    # Mock LLM client
    sm._llm_client = AsyncMock()

    # Mock thread
    sm._thread = MagicMock()
    sm._thread.messages = []
    sm._thread.to_checkpoint.return_value = None

    # Mock policy enforcer and token budget
    sm._policy_enforcer = MagicMock()
    sm._token_budget = MagicMock()
    sm._token_budget.input_tokens_used = 0
    sm._token_budget.output_tokens_used = 0

    # Mock workspace client
    sm._workspace_client = AsyncMock()

    # Mock checkpoint manager
    sm._checkpoint_manager = MagicMock()

    return sm


class TestRunAgentExceptionHandling:
    @pytest.mark.asyncio
    async def test_policy_expired_emits_session_failed(self) -> None:
        """PolicyExpiredError triggers emit_session_failed (session-level)."""
        sm = _make_session_manager()

        # Make AgentLoop.run raise PolicyExpiredError
        with patch("agent_host.session.session_manager.AgentLoop") as mock_loop_cls:
            mock_loop = AsyncMock()
            mock_loop.run.side_effect = PolicyExpiredError("Policy has expired")
            mock_loop_cls.return_value = mock_loop

            await sm._run_agent("test prompt", "task-1", 50)

        sm._event_emitter.emit_session_failed.assert_called_once_with("Policy has expired")
        sm._event_emitter.emit_task_failed.assert_not_called()

    @pytest.mark.asyncio
    async def test_checkpoint_error_emits_session_failed(self) -> None:
        """CheckpointError triggers emit_session_failed (session-level)."""
        sm = _make_session_manager()

        with patch("agent_host.session.session_manager.AgentLoop") as mock_loop_cls:
            mock_loop = AsyncMock()
            mock_loop.run.side_effect = CheckpointError("Checkpoint corrupt")
            mock_loop_cls.return_value = mock_loop

            await sm._run_agent("test prompt", "task-1", 50)

        sm._event_emitter.emit_session_failed.assert_called_once_with("Checkpoint corrupt")

    @pytest.mark.asyncio
    async def test_transient_llm_error_emits_enriched_task_failed(self) -> None:
        """A transient LLM error (e.g. 429) passes enriched params to emit_task_failed."""
        sm = _make_session_manager()

        # Create a fake 429 error
        class FakeRateLimitError(Exception):
            status_code = 429

        with patch("agent_host.session.session_manager.AgentLoop") as mock_loop_cls:
            mock_loop = AsyncMock()
            mock_loop.run.side_effect = FakeRateLimitError("rate limited")
            mock_loop_cls.return_value = mock_loop

            await sm._run_agent("test prompt", "task-1", 50)

        sm._event_emitter.emit_task_failed.assert_called_once()
        call_kwargs = sm._event_emitter.emit_task_failed.call_args
        assert call_kwargs[0][0] == "task-1"  # positional: task_id
        assert call_kwargs.kwargs["error_code"] == "RATE_LIMITED"
        assert call_kwargs.kwargs["error_type"] == "rate_limit"
        assert call_kwargs.kwargs["is_recoverable"] is True
        assert "rate limited" in call_kwargs.kwargs["reason"].lower()

    @pytest.mark.asyncio
    async def test_overloaded_529_emits_enriched_task_failed(self) -> None:
        """A 529 overloaded error passes LLM_OVERLOADED to emit_task_failed."""
        sm = _make_session_manager()

        class FakeOverloadedError(Exception):
            status_code = 529

        with patch("agent_host.session.session_manager.AgentLoop") as mock_loop_cls:
            mock_loop = AsyncMock()
            mock_loop.run.side_effect = FakeOverloadedError("overloaded")
            mock_loop_cls.return_value = mock_loop

            await sm._run_agent("test prompt", "task-1", 50)

        call_kwargs = sm._event_emitter.emit_task_failed.call_args
        assert call_kwargs.kwargs["error_code"] == "LLM_OVERLOADED"
        assert call_kwargs.kwargs["error_type"] == "overloaded"
        assert call_kwargs.kwargs["is_recoverable"] is True

    @pytest.mark.asyncio
    async def test_non_llm_error_emits_plain_task_failed(self) -> None:
        """A non-transient error (e.g. ValueError) emits task_failed without LLM enrichment."""
        sm = _make_session_manager()

        with patch("agent_host.session.session_manager.AgentLoop") as mock_loop_cls:
            mock_loop = AsyncMock()
            mock_loop.run.side_effect = ValueError("bad value")
            mock_loop_cls.return_value = mock_loop

            await sm._run_agent("test prompt", "task-1", 50)

        call_kwargs = sm._event_emitter.emit_task_failed.call_args
        assert call_kwargs[0][0] == "task-1"
        assert "bad value" in call_kwargs.kwargs["reason"]
        # Non-LLM errors are classified as permanent LLM_ERROR
        assert call_kwargs.kwargs["error_code"] == "LLM_ERROR"
        assert call_kwargs.kwargs["error_type"] == "permanent"
        assert call_kwargs.kwargs["is_recoverable"] is False

    @pytest.mark.asyncio
    async def test_cancelled_error_does_not_emit(self) -> None:
        """asyncio.CancelledError should not emit any failure events."""
        sm = _make_session_manager()

        with patch("agent_host.session.session_manager.AgentLoop") as mock_loop_cls:
            mock_loop = AsyncMock()
            mock_loop.run.side_effect = asyncio.CancelledError()
            mock_loop_cls.return_value = mock_loop

            await sm._run_agent("test prompt", "task-1", 50)

        sm._event_emitter.emit_task_failed.assert_not_called()
        sm._event_emitter.emit_session_failed.assert_not_called()
