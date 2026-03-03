"""LLM error classification — transient vs permanent.

Used by the retry loop to decide whether to retry a failed LLM call.
Uses duck-typing and class name matching to avoid hard litellm import.
"""

from __future__ import annotations

import httpx

from agent_host.exceptions import (
    LLMBudgetExceededError,
    LLMGatewayError,
    LLMGuardrailBlockedError,
    PolicyExpiredError,
)

# HTTP status codes that indicate transient server-side issues
# 529 = Anthropic "Overloaded" (server too busy, retry later)
_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 502, 503, 504, 529})

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
    return class_name in _TRANSIENT_CLASS_NAMES


def classify_llm_error(exc: Exception) -> dict[str, str | bool]:
    """Classify an LLM error into structured metadata for event payloads.

    Handles both raw SDK exceptions and wrapped ``LLMGatewayError`` (which
    the LLM client produces after exhausting retries).  For wrapped errors
    it inspects ``details["error_type"]`` and the chained ``__cause__``.

    Returns a dict with keys:
        error_code: e.g. "RATE_LIMITED", "LLM_OVERLOADED", "LLM_UNAVAILABLE", "LLM_ERROR"
        error_type: "rate_limit", "overloaded", "transient", "permanent"
        is_recoverable: bool
        user_message: clean user-facing string
    """
    # --- Wrapped LLMGatewayError (retry-exhausted) ---
    # The LLM client wraps transient errors in LLMGatewayError with
    # details={"error_type": "rate_limit"|"transient", "original_error": ...}.
    # Classify via the original __cause__ first, then fall back to details.
    if isinstance(exc, LLMGatewayError):
        cause = exc.__cause__
        if cause is not None and isinstance(cause, Exception):
            return classify_llm_error(cause)
        # No chained cause — inspect details dict
        err_type = exc.details.get("error_type", "")
        if err_type == "rate_limit":
            return {
                "error_code": "RATE_LIMITED",
                "error_type": "rate_limit",
                "is_recoverable": True,
                "user_message": (
                    "Rate limited by the LLM provider. Please wait a moment and try again."
                ),
            }
        # Any other LLMGatewayError is a transient failure (retries exhausted)
        return {
            "error_code": "LLM_UNAVAILABLE",
            "error_type": "transient",
            "is_recoverable": True,
            "user_message": str(exc),
        }

    # --- Raw SDK / transport exceptions ---
    status_code = getattr(exc, "status_code", None)

    # Rate limit (429)
    if status_code == 429 or type(exc).__name__ == "RateLimitError":
        return {
            "error_code": "RATE_LIMITED",
            "error_type": "rate_limit",
            "is_recoverable": True,
            "user_message": "Rate limited by the LLM provider. Please wait a moment and try again.",
        }

    # Overloaded (529 — Anthropic-specific)
    if status_code == 529:
        return {
            "error_code": "LLM_OVERLOADED",
            "error_type": "overloaded",
            "is_recoverable": True,
            "user_message": "The LLM service is overloaded. Please try again shortly.",
        }

    # Other transient errors (502, 503, 504, connection errors, timeouts)
    if is_transient_llm_error(exc):
        return {
            "error_code": "LLM_UNAVAILABLE",
            "error_type": "transient",
            "is_recoverable": True,
            "user_message": "LLM service is temporarily unavailable. Please try again.",
        }

    # Permanent / unknown errors
    return {
        "error_code": "LLM_ERROR",
        "error_type": "permanent",
        "is_recoverable": False,
        "user_message": str(exc) or "An unexpected LLM error occurred.",
    }


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if the exception is specifically a rate limit error (429)."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    return type(exc).__name__ == "RateLimitError"


def extract_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After delay from a rate limit error, if available.

    OpenAI SDK exceptions carry the original httpx.Response in `exc.response`,
    which may include a Retry-After header.

    Returns:
        Delay in seconds, or None if not available.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None

    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return float(retry_after)
    except (ValueError, TypeError):
        return None
