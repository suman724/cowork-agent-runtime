"""LLM error classification — transient vs permanent.

Used by the retry loop to decide whether to retry a failed LLM call.
Uses duck-typing and class name matching to avoid hard litellm import.
"""

from __future__ import annotations

import httpx

from agent_host.exceptions import (
    LLMBudgetExceededError,
    LLMGuardrailBlockedError,
    PolicyExpiredError,
)

# HTTP status codes that indicate transient server-side issues
_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 502, 503, 504})

# Exception class names (from litellm/openai SDKs) that indicate transient errors
_TRANSIENT_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "RateLimitError",
        "ServiceUnavailableError",
        "APIConnectionError",
        "Timeout",
        "APITimeoutError",
    }
)

# Exceptions that are never retryable — session-level or permanent failures
_PERMANENT_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    LLMBudgetExceededError,
    LLMGuardrailBlockedError,
    PolicyExpiredError,
)


def is_transient_llm_error(exc: Exception) -> bool:
    """Classify whether an LLM-related exception is transient (retryable).

    Returns True for transient errors (network failures, rate limits, 5xx),
    False for permanent errors (budget exceeded, guardrail blocked, policy expired).
    """
    # Permanent errors are never retryable
    if isinstance(exc, _PERMANENT_EXCEPTION_TYPES):
        return False

    # httpx transport-level errors are always transient
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException)):
        return True

    # Python built-in connection errors are transient
    if isinstance(exc, (ConnectionError, OSError)):
        return True

    # Check for HTTP status code via duck-typing (litellm, openai, httpx.HTTPStatusError)
    status_code = getattr(exc, "status_code", None)
    if status_code is not None and status_code in _TRANSIENT_STATUS_CODES:
        return True

    # Check class name for known SDK exception types
    class_name = type(exc).__name__
    if class_name in _TRANSIENT_CLASS_NAMES:
        return True

    # Unknown exceptions are not retryable by default
    return False
