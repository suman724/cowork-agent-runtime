"""Tests for SessionManager — _run_agent exception handling with enriched error events."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_host.events.event_emitter import EventEmitter
from agent_host.exceptions import CheckpointError, PolicyExpiredError
from agent_host.models import SessionContext
from agent_host.session.checkpoint_manager import SessionCheckpoint
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
    config.workspace_sync_interval = 5

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


class TestOnStepComplete:
    @pytest.mark.asyncio
    async def test_per_step_checkpoint_saves(self) -> None:
        """_on_step_complete should call _persist_checkpoint with active task state."""
        sm = _make_session_manager()
        sm._current_task_prompt = "Fix the bug"
        sm._current_max_steps = 50

        await sm._on_step_complete("task-1", 3)

        sm._checkpoint_manager.save.assert_called()
        saved_checkpoint = sm._checkpoint_manager.save.call_args[0][0]
        assert saved_checkpoint.active_task_id == "task-1"
        assert saved_checkpoint.active_task_prompt == "Fix the bug"
        assert saved_checkpoint.active_task_step == 3
        assert saved_checkpoint.active_task_max_steps == 50

    @pytest.mark.asyncio
    async def test_periodic_workspace_sync(self) -> None:
        """_on_step_complete should sync to workspace at sync_interval steps."""
        sm = _make_session_manager()
        sm._config.workspace_sync_interval = 3
        sm._current_task_prompt = "Test"
        sm._current_max_steps = 50

        # Steps 1 and 2 should not trigger sync
        await sm._on_step_complete("task-1", 1)
        await sm._on_step_complete("task-1", 2)
        sm._workspace_client.upload_session_history.assert_not_called()

        # Step 3 should trigger sync (3 - 0 >= 3)
        await sm._on_step_complete("task-1", 3)
        sm._workspace_client.upload_session_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_workspace_sync_disabled_when_interval_zero(self) -> None:
        """No workspace sync when workspace_sync_interval is 0."""
        sm = _make_session_manager()
        sm._config.workspace_sync_interval = 0
        sm._current_task_prompt = "Test"
        sm._current_max_steps = 50

        await sm._on_step_complete("task-1", 5)
        await sm._on_step_complete("task-1", 10)
        sm._workspace_client.upload_session_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_checkpoint_saved_event_emitted(self) -> None:
        """_on_step_complete should emit checkpoint_saved event."""
        sm = _make_session_manager()
        sm._current_task_prompt = "Test"
        sm._current_max_steps = 50

        await sm._on_step_complete("task-1", 1)
        sm._event_emitter.emit_checkpoint_saved.assert_called_once_with("task-1", 1)


class TestIncompleteTaskDetection:
    @pytest.mark.asyncio
    async def test_incomplete_task_detected_on_restore(self) -> None:
        """_restore_from_checkpoint should detect an in-progress task."""
        sm = _make_session_manager()
        sm._checkpoint_manager.load.return_value = SessionCheckpoint(
            session_id="sess-1",
            workspace_id="ws-1",
            tenant_id="t-1",
            user_id="u-1",
            active_task_id="task-42",
            active_task_prompt="Fix the bug",
            active_task_step=7,
            active_task_max_steps=50,
        )

        await sm._restore_from_checkpoint()

        assert sm._incomplete_task is not None
        assert sm._incomplete_task["taskId"] == "task-42"
        assert sm._incomplete_task["prompt"] == "Fix the bug"
        assert sm._incomplete_task["lastStep"] == 7
        assert sm._incomplete_task["maxSteps"] == 50

    @pytest.mark.asyncio
    async def test_no_incomplete_task_when_none(self) -> None:
        """_restore_from_checkpoint should not set _incomplete_task when active_task_id is None."""
        sm = _make_session_manager()
        sm._checkpoint_manager.load.return_value = SessionCheckpoint(
            session_id="sess-1",
            workspace_id="ws-1",
            tenant_id="t-1",
            user_id="u-1",
            active_task_id=None,
        )

        await sm._restore_from_checkpoint()

        assert sm._incomplete_task is None

    @pytest.mark.asyncio
    async def test_incomplete_task_in_get_session_state(self) -> None:
        """get_session_state should include incompleteTask when set."""
        sm = _make_session_manager()
        sm._incomplete_task = {
            "taskId": "task-42",
            "prompt": "Fix the bug",
            "lastStep": 7,
            "maxSteps": 50,
        }

        state = await sm.get_session_state()
        assert "incompleteTask" in state
        assert state["incompleteTask"]["taskId"] == "task-42"

    @pytest.mark.asyncio
    async def test_no_incomplete_task_in_get_session_state(self) -> None:
        """get_session_state should not include incompleteTask when not set."""
        sm = _make_session_manager()
        sm._incomplete_task = None

        state = await sm.get_session_state()
        assert "incompleteTask" not in state


class TestResumeIncompleteTask:
    @pytest.mark.asyncio
    async def test_resume_skips_add_user_message(self) -> None:
        """start_task with the same taskId as incomplete should skip add_user_message."""
        sm = _make_session_manager()
        sm._incomplete_task = {
            "taskId": "task-42",
            "prompt": "Fix the bug",
            "lastStep": 7,
            "maxSteps": 50,
        }

        result = await sm.start_task(
            {
                "taskId": "task-42",
                "prompt": "Fix the bug",
            }
        )

        assert result["status"] == "running"
        # Thread.add_user_message should NOT have been called
        sm._thread.add_user_message.assert_not_called()
        # _incomplete_task should be cleared after resume
        assert sm._incomplete_task is None

    @pytest.mark.asyncio
    async def test_non_resume_adds_user_message(self) -> None:
        """start_task with a different taskId should add_user_message normally."""
        sm = _make_session_manager()
        sm._incomplete_task = {
            "taskId": "task-42",
            "prompt": "Fix the bug",
            "lastStep": 7,
            "maxSteps": 50,
        }

        result = await sm.start_task(
            {
                "taskId": "task-99",
                "prompt": "New task",
            }
        )

        assert result["status"] == "running"
        sm._thread.add_user_message.assert_called_once()


class TestWorkspaceDirCheckpoint:
    @pytest.mark.asyncio
    async def test_resume_restores_workspace_dir_from_checkpoint(self) -> None:
        """_restore_from_checkpoint should restore workspace_dir when missing."""
        sm = _make_session_manager()
        sm._workspace_dir = None  # Simulate lost workspace_dir on resume

        sm._checkpoint_manager.load.return_value = SessionCheckpoint(
            session_id="sess-1",
            workspace_id="ws-1",
            tenant_id="t-1",
            user_id="u-1",
            workspace_dir="/home/user/project",
        )

        await sm._restore_from_checkpoint()

        assert sm._workspace_dir == "/home/user/project"

    @pytest.mark.asyncio
    async def test_workspace_dir_not_overwritten_if_already_set(self) -> None:
        """_restore_from_checkpoint should not overwrite existing workspace_dir."""
        sm = _make_session_manager()
        sm._workspace_dir = "/current/workspace"

        sm._checkpoint_manager.load.return_value = SessionCheckpoint(
            session_id="sess-1",
            workspace_id="ws-1",
            tenant_id="t-1",
            user_id="u-1",
            workspace_dir="/old/workspace",
        )

        await sm._restore_from_checkpoint()

        # Should keep the current one, not overwrite with checkpoint value
        assert sm._workspace_dir == "/current/workspace"

    @pytest.mark.asyncio
    async def test_persist_checkpoint_includes_workspace_dir(self) -> None:
        """_persist_checkpoint should include workspace_dir in the saved checkpoint."""
        sm = _make_session_manager()
        sm._workspace_dir = "/home/user/project"

        sm._persist_checkpoint()

        sm._checkpoint_manager.save.assert_called()
        saved_checkpoint = sm._checkpoint_manager.save.call_args[0][0]
        assert saved_checkpoint.workspace_dir == "/home/user/project"


class TestWorkspaceFallback:
    @pytest.mark.asyncio
    async def test_workspace_fallback_on_missing_checkpoint(self) -> None:
        """When local checkpoint is None, fall back to workspace service."""
        sm = _make_session_manager()
        sm._checkpoint_manager.load.return_value = None

        from datetime import UTC, datetime

        from cowork_platform.conversation_message import ConversationMessage

        mock_messages = [
            ConversationMessage(
                messageId="m1",
                sessionId="sess-1",
                taskId="task-1",
                role="user",
                content="Hello",
                timestamp=datetime.now(tz=UTC),
            ),
        ]
        sm._workspace_client.get_session_history = AsyncMock(return_value=mock_messages)

        await sm._restore_from_checkpoint()

        sm._workspace_client.get_session_history.assert_called_once_with(
            workspace_id="ws-1",
            session_id="sess-1",
        )
        assert len(sm._session_messages) == 1
        assert sm._session_messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_workspace_fallback_failure_is_silent(self) -> None:
        """Workspace fallback failure should be logged but not raise."""
        sm = _make_session_manager()
        sm._checkpoint_manager.load.return_value = None
        sm._workspace_client.get_session_history = AsyncMock(side_effect=Exception("network error"))

        # Should not raise
        await sm._restore_from_checkpoint()
        assert sm._session_messages == []
