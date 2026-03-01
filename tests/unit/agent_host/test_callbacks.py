"""Tests for ADK callbacks — policy enforcement, budget, events."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_host.agent.callbacks import (
    make_after_tool_callback,
    make_before_model_callback,
    make_before_tool_callback,
)
from agent_host.budget.token_budget import TokenBudget
from agent_host.exceptions import LLMBudgetExceededError, PolicyExpiredError
from agent_host.policy.policy_enforcer import PolicyEnforcer
from tests.fixtures.policy_bundles import (
    make_policy_bundle,
    make_restrictive_bundle,
)


class TestBeforeToolCallback:
    def test_allowed_tool_returns_none(self) -> None:
        """Allowed tool calls return None (proceed)."""
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        callback = make_before_tool_callback(enforcer)

        result = callback(MagicMock(), "ReadFile", {"path": "/tmp/test"})
        assert result is None

    def test_denied_tool_returns_dict(self) -> None:
        """Denied tool calls return error dict (skip execution)."""
        bundle = make_policy_bundle(
            capabilities=[{"name": "LLM.Call"}]  # No File.Read
        )
        enforcer = PolicyEnforcer(bundle)
        callback = make_before_tool_callback(enforcer)

        result = callback(MagicMock(), "ReadFile", {"path": "/tmp/test"})
        assert result is not None
        assert result["status"] == "denied"
        assert result["error"]["code"] == "CAPABILITY_DENIED"

    def test_denied_by_path_restriction(self) -> None:
        """Path restriction denies tool call."""
        bundle = make_restrictive_bundle(allowed_paths=["/allowed"])
        enforcer = PolicyEnforcer(bundle)
        callback = make_before_tool_callback(enforcer)

        result = callback(MagicMock(), "ReadFile", {"path": "/forbidden/file"})
        assert result is not None
        assert result["status"] == "denied"

    def test_unknown_tool_passes_through(self) -> None:
        """Unknown tools (not in capability map) pass through."""
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        callback = make_before_tool_callback(enforcer)

        result = callback(MagicMock(), "google_search", {"query": "test"})
        assert result is None

    def test_approval_required_passes_through(self) -> None:
        """Tools requiring approval return None (LongRunningFunctionTool handles it)."""
        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        callback = make_before_tool_callback(enforcer)

        result = callback(MagicMock(), "ReadFile", {"path": "/tmp/x"})
        assert result is None

    def test_emits_tool_requested_event(self) -> None:
        """Event emitter is called when tool proceeds."""
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        emitter = MagicMock()
        callback = make_before_tool_callback(enforcer, event_emitter=emitter)

        callback(MagicMock(), "ReadFile", {"path": "/tmp/test"})
        emitter.emit_tool_requested.assert_called_once()

    def test_emits_approval_requested_event(self) -> None:
        """Emits approval_requested event when tool requires approval."""
        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        emitter = MagicMock()
        callback = make_before_tool_callback(enforcer, event_emitter=emitter)

        callback(MagicMock(), "ReadFile", {"path": "/tmp/x"})
        emitter.emit_approval_requested.assert_called_once()
        call_kwargs = emitter.emit_approval_requested.call_args
        assert call_kwargs[1]["tool_name"] == "ReadFile"


class TestBeforeModelCallback:
    def test_passes_when_allowed(self) -> None:
        """Returns None when LLM call is allowed and budget is available."""
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        budget = TokenBudget(max_session_tokens=100000)
        callback = make_before_model_callback(enforcer, budget)

        ctx = MagicMock()
        ctx.state = {}
        result = callback(ctx, MagicMock())
        assert result is None

    def test_raises_policy_expired(self) -> None:
        """Raises PolicyExpiredError when policy is expired."""
        bundle = make_policy_bundle(expired=True)
        enforcer = PolicyEnforcer(bundle)
        budget = TokenBudget(max_session_tokens=100000)
        callback = make_before_model_callback(enforcer, budget)

        with pytest.raises(PolicyExpiredError):
            callback(MagicMock(), MagicMock())

    def test_emits_policy_expired_event(self) -> None:
        """Emits policy_expired event when policy is expired."""
        bundle = make_policy_bundle(expired=True)
        enforcer = PolicyEnforcer(bundle)
        budget = TokenBudget(max_session_tokens=100000)
        emitter = MagicMock()
        callback = make_before_model_callback(enforcer, budget, event_emitter=emitter)

        with pytest.raises(PolicyExpiredError):
            callback(MagicMock(), MagicMock())

        emitter.emit_policy_expired.assert_called_once()

    def test_raises_budget_exceeded(self) -> None:
        """Raises LLMBudgetExceededError when budget exhausted."""
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        budget = TokenBudget(max_session_tokens=100)
        budget.record_usage(input_tokens=100, output_tokens=0)
        callback = make_before_model_callback(enforcer, budget)

        with pytest.raises(LLMBudgetExceededError):
            callback(MagicMock(), MagicMock())


class TestAfterToolCallback:
    def test_returns_none(self) -> None:
        """After tool callback returns None (use result as-is)."""
        callback = make_after_tool_callback()
        result = callback(MagicMock(), "ReadFile", {"status": "succeeded"})
        assert result is None

    def test_emits_tool_completed(self) -> None:
        """Event emitter is called with tool completion."""
        emitter = MagicMock()
        callback = make_after_tool_callback(event_emitter=emitter)
        callback(MagicMock(), "ReadFile", {"status": "succeeded"})
        emitter.emit_tool_completed.assert_called_once_with("ReadFile", "succeeded")
