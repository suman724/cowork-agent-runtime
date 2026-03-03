"""Tests for LLM error classifier — classify_llm_error() + is_transient_llm_error()."""

from __future__ import annotations

import httpx

from agent_host.exceptions import LLMBudgetExceededError, LLMGatewayError, LLMGuardrailBlockedError
from agent_host.llm.error_classifier import (
    classify_llm_error,
    extract_retry_after,
    is_rate_limit_error,
    is_transient_llm_error,
)


class _FakeStatusError(Exception):
    """Simulates an SDK error with a status_code attribute."""

    def __init__(self, status_code: int, message: str = "error") -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeRateLimitError(Exception):
    """Simulates an OpenAI/litellm RateLimitError by class name."""


# Rename to match SDK class name for duck-typing
_FakeRateLimitError.__name__ = "RateLimitError"
_FakeRateLimitError.__qualname__ = "RateLimitError"


class TestClassifyLlmError:
    def test_rate_limit_429(self) -> None:
        exc = _FakeStatusError(429)
        result = classify_llm_error(exc)
        assert result["error_code"] == "RATE_LIMITED"
        assert result["error_type"] == "rate_limit"
        assert result["is_recoverable"] is True
        assert "rate limited" in str(result["user_message"]).lower()

    def test_rate_limit_by_class_name(self) -> None:
        exc = _FakeRateLimitError("Too many requests")
        result = classify_llm_error(exc)
        assert result["error_code"] == "RATE_LIMITED"
        assert result["error_type"] == "rate_limit"
        assert result["is_recoverable"] is True

    def test_overloaded_529(self) -> None:
        exc = _FakeStatusError(529)
        result = classify_llm_error(exc)
        assert result["error_code"] == "LLM_OVERLOADED"
        assert result["error_type"] == "overloaded"
        assert result["is_recoverable"] is True
        assert "overloaded" in str(result["user_message"]).lower()

    def test_transient_502(self) -> None:
        exc = _FakeStatusError(502)
        result = classify_llm_error(exc)
        assert result["error_code"] == "LLM_UNAVAILABLE"
        assert result["error_type"] == "transient"
        assert result["is_recoverable"] is True

    def test_transient_503(self) -> None:
        exc = _FakeStatusError(503)
        result = classify_llm_error(exc)
        assert result["error_code"] == "LLM_UNAVAILABLE"
        assert result["error_type"] == "transient"

    def test_transient_504(self) -> None:
        exc = _FakeStatusError(504)
        result = classify_llm_error(exc)
        assert result["error_code"] == "LLM_UNAVAILABLE"
        assert result["error_type"] == "transient"

    def test_transient_connection_error(self) -> None:
        exc = httpx.ConnectError("Connection refused")
        result = classify_llm_error(exc)
        assert result["error_code"] == "LLM_UNAVAILABLE"
        assert result["error_type"] == "transient"
        assert result["is_recoverable"] is True

    def test_transient_timeout(self) -> None:
        exc = httpx.ReadTimeout("Read timed out")
        result = classify_llm_error(exc)
        assert result["error_code"] == "LLM_UNAVAILABLE"
        assert result["error_type"] == "transient"
        assert result["is_recoverable"] is True

    def test_permanent_unknown_error(self) -> None:
        exc = ValueError("bad input")
        result = classify_llm_error(exc)
        assert result["error_code"] == "LLM_ERROR"
        assert result["error_type"] == "permanent"
        assert result["is_recoverable"] is False
        assert "bad input" in str(result["user_message"])

    def test_permanent_unknown_http_error(self) -> None:
        """Non-transient HTTP status (e.g. 400) should be permanent."""
        exc = _FakeStatusError(400, "Bad request")
        result = classify_llm_error(exc)
        assert result["error_code"] == "LLM_ERROR"
        assert result["error_type"] == "permanent"
        assert result["is_recoverable"] is False

    def test_wrapped_llm_gateway_error_with_529_cause(self) -> None:
        """LLMGatewayError wrapping a 529 should classify as overloaded."""
        raw = _FakeStatusError(529, "Overloaded")
        wrapped = LLMGatewayError(
            "LLM service is temporarily unavailable.",
            details={"error_type": "transient", "original_error": str(raw)},
        )
        wrapped.__cause__ = raw
        result = classify_llm_error(wrapped)
        assert result["error_code"] == "LLM_OVERLOADED"
        assert result["error_type"] == "overloaded"
        assert result["is_recoverable"] is True

    def test_wrapped_llm_gateway_error_with_429_cause(self) -> None:
        """LLMGatewayError wrapping a 429 should classify as rate limited."""
        raw = _FakeStatusError(429, "Too Many Requests")
        wrapped = LLMGatewayError(
            "Rate limited by the LLM provider.",
            details={"error_type": "rate_limit", "original_error": str(raw)},
        )
        wrapped.__cause__ = raw
        result = classify_llm_error(wrapped)
        assert result["error_code"] == "RATE_LIMITED"
        assert result["error_type"] == "rate_limit"
        assert result["is_recoverable"] is True

    def test_wrapped_llm_gateway_error_no_cause_rate_limit(self) -> None:
        """LLMGatewayError with rate_limit details but no __cause__."""
        wrapped = LLMGatewayError(
            "Rate limited.",
            details={"error_type": "rate_limit"},
        )
        result = classify_llm_error(wrapped)
        assert result["error_code"] == "RATE_LIMITED"
        assert result["error_type"] == "rate_limit"

    def test_wrapped_llm_gateway_error_no_cause_transient(self) -> None:
        """LLMGatewayError with transient details but no __cause__."""
        wrapped = LLMGatewayError(
            "LLM service is temporarily unavailable. Please try again.",
            details={"error_type": "transient"},
        )
        result = classify_llm_error(wrapped)
        assert result["error_code"] == "LLM_UNAVAILABLE"
        assert result["error_type"] == "transient"
        assert result["is_recoverable"] is True


class TestIsTransientLlmError:
    def test_budget_exceeded_not_transient(self) -> None:
        exc = LLMBudgetExceededError("over budget")
        assert is_transient_llm_error(exc) is False

    def test_guardrail_blocked_not_transient(self) -> None:
        exc = LLMGuardrailBlockedError("blocked")
        assert is_transient_llm_error(exc) is False

    def test_rate_limit_is_transient(self) -> None:
        exc = _FakeStatusError(429)
        assert is_transient_llm_error(exc) is True

    def test_connection_error_is_transient(self) -> None:
        exc = ConnectionError("refused")
        assert is_transient_llm_error(exc) is True


class TestIsRateLimitError:
    def test_429_status(self) -> None:
        exc = _FakeStatusError(429)
        assert is_rate_limit_error(exc) is True

    def test_non_429_status(self) -> None:
        exc = _FakeStatusError(503)
        assert is_rate_limit_error(exc) is False

    def test_class_name_match(self) -> None:
        exc = _FakeRateLimitError("limit")
        assert is_rate_limit_error(exc) is True


class TestExtractRetryAfter:
    def test_no_response(self) -> None:
        assert extract_retry_after(ValueError("x")) is None

    def test_with_retry_after_header(self) -> None:
        class FakeResp:
            headers: dict[str, str] = {"retry-after": "2.5"}  # noqa: RUF012

        class FakeRetryError(Exception):
            response = FakeResp()

        assert extract_retry_after(FakeRetryError()) == 2.5
