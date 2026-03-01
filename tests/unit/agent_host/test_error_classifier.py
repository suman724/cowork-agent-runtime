"""Tests for LLM error classifier — transient vs permanent."""

from __future__ import annotations

import httpx
import pytest

from agent_host.exceptions import (
    LLMBudgetExceededError,
    LLMGuardrailBlockedError,
    PolicyExpiredError,
)
from agent_host.llm.error_classifier import is_transient_llm_error


class TestTransientErrors:
    def test_httpx_connect_error(self) -> None:
        exc = httpx.ConnectError("Connection refused")
        assert is_transient_llm_error(exc) is True

    def test_httpx_read_error(self) -> None:
        exc = httpx.ReadError("Connection reset")
        assert is_transient_llm_error(exc) is True

    def test_httpx_timeout(self) -> None:
        exc = httpx.ReadTimeout("Read timed out")
        assert is_transient_llm_error(exc) is True

    def test_connection_error(self) -> None:
        exc = ConnectionError("Connection refused")
        assert is_transient_llm_error(exc) is True

    def test_os_error(self) -> None:
        exc = OSError("Network is unreachable")
        assert is_transient_llm_error(exc) is True

    @pytest.mark.parametrize("status_code", [429, 502, 503, 504])
    def test_transient_status_codes(self, status_code: int) -> None:
        """Exceptions with transient HTTP status codes are retryable."""
        exc = Exception("Server error")
        exc.status_code = status_code  # type: ignore[attr-defined]
        assert is_transient_llm_error(exc) is True

    def test_rate_limit_error_by_class_name(self) -> None:
        """Exception class named RateLimitError is transient."""

        class RateLimitError(Exception):
            pass

        assert is_transient_llm_error(RateLimitError("Too many requests")) is True

    def test_api_connection_error_by_class_name(self) -> None:
        """Exception class named APIConnectionError is transient."""

        class APIConnectionError(Exception):
            pass

        assert is_transient_llm_error(APIConnectionError("Connection failed")) is True


class TestPermanentErrors:
    def test_budget_exceeded(self) -> None:
        exc = LLMBudgetExceededError("Budget exhausted")
        assert is_transient_llm_error(exc) is False

    def test_guardrail_blocked(self) -> None:
        exc = LLMGuardrailBlockedError("Content blocked")
        assert is_transient_llm_error(exc) is False

    def test_policy_expired(self) -> None:
        exc = PolicyExpiredError("Policy expired")
        assert is_transient_llm_error(exc) is False

    def test_generic_exception(self) -> None:
        exc = ValueError("Something went wrong")
        assert is_transient_llm_error(exc) is False

    def test_non_transient_status_code(self) -> None:
        """Exceptions with 400/401/403 status codes are not transient."""
        exc = Exception("Bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        assert is_transient_llm_error(exc) is False

    def test_runtime_error(self) -> None:
        exc = RuntimeError("Unexpected error")
        assert is_transient_llm_error(exc) is False
