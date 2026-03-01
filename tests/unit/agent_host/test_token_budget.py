"""Tests for TokenBudget — session-level token tracking."""

from __future__ import annotations

import pytest

from agent_host.budget.token_budget import TokenBudget
from agent_host.exceptions import LLMBudgetExceededError


class TestTokenBudget:
    def test_initial_state(self) -> None:
        budget = TokenBudget(max_session_tokens=10000)
        assert budget.max_session_tokens == 10000
        assert budget.total_tokens_used == 0
        assert budget.input_tokens_used == 0
        assert budget.output_tokens_used == 0
        assert budget.remaining == 10000

    def test_record_usage(self) -> None:
        budget = TokenBudget(max_session_tokens=10000)
        budget.record_usage(input_tokens=500, output_tokens=200)
        assert budget.input_tokens_used == 500
        assert budget.output_tokens_used == 200
        assert budget.total_tokens_used == 700
        assert budget.remaining == 9300

    def test_cumulative_usage(self) -> None:
        budget = TokenBudget(max_session_tokens=10000)
        budget.record_usage(input_tokens=500, output_tokens=200)
        budget.record_usage(input_tokens=300, output_tokens=100)
        assert budget.total_tokens_used == 1100
        assert budget.remaining == 8900

    def test_pre_check_passes_within_budget(self) -> None:
        budget = TokenBudget(max_session_tokens=10000)
        budget.record_usage(input_tokens=5000, output_tokens=2000)
        budget.pre_check()  # Should not raise

    def test_pre_check_passes_with_estimate(self) -> None:
        budget = TokenBudget(max_session_tokens=10000)
        budget.record_usage(input_tokens=5000, output_tokens=2000)
        budget.pre_check(estimated_tokens=2000)  # Should not raise

    def test_pre_check_fails_when_exhausted(self) -> None:
        budget = TokenBudget(max_session_tokens=1000)
        budget.record_usage(input_tokens=600, output_tokens=400)
        with pytest.raises(LLMBudgetExceededError, match="exhausted"):
            budget.pre_check()

    def test_pre_check_fails_when_estimate_exceeds(self) -> None:
        budget = TokenBudget(max_session_tokens=1000)
        budget.record_usage(input_tokens=400, output_tokens=200)
        with pytest.raises(LLMBudgetExceededError, match="would exceed"):
            budget.pre_check(estimated_tokens=500)

    def test_pre_check_zero_estimate_ignored(self) -> None:
        budget = TokenBudget(max_session_tokens=1000)
        budget.record_usage(input_tokens=400, output_tokens=200)
        budget.pre_check(estimated_tokens=0)  # Should not raise

    def test_remaining_never_negative(self) -> None:
        budget = TokenBudget(max_session_tokens=100)
        budget.record_usage(input_tokens=200, output_tokens=50)
        assert budget.remaining == 0

    def test_error_details(self) -> None:
        budget = TokenBudget(max_session_tokens=1000)
        budget.record_usage(input_tokens=1000, output_tokens=0)
        with pytest.raises(LLMBudgetExceededError) as exc_info:
            budget.pre_check()
        assert exc_info.value.details["total_used"] == 1000
        assert exc_info.value.details["max_allowed"] == 1000

    def test_restore_usage(self) -> None:
        """restore_usage sets absolute values (not additive)."""
        budget = TokenBudget(max_session_tokens=10000)
        budget.record_usage(input_tokens=100, output_tokens=50)
        budget.restore_usage(input_tokens=500, output_tokens=200)
        assert budget.input_tokens_used == 500
        assert budget.output_tokens_used == 200
        assert budget.total_tokens_used == 700
        assert budget.remaining == 9300

    def test_restore_then_record(self) -> None:
        """restore_usage followed by record_usage is cumulative."""
        budget = TokenBudget(max_session_tokens=10000)
        budget.restore_usage(input_tokens=500, output_tokens=200)
        budget.record_usage(input_tokens=100, output_tokens=50)
        assert budget.input_tokens_used == 600
        assert budget.output_tokens_used == 250
        assert budget.total_tokens_used == 850
        assert budget.remaining == 9150
