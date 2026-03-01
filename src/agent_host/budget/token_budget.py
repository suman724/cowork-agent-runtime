"""Session-level token budget tracking.

Used in before_model_callback to prevent overspending the session's
token budget. Actual counts come from LLM response usage metadata.
"""

from __future__ import annotations

from agent_host.exceptions import LLMBudgetExceededError


class TokenBudget:
    """Tracks cumulative token usage against a session budget.

    Thread-safe for single-threaded asyncio (no locks needed).
    """

    def __init__(self, max_session_tokens: int) -> None:
        self._max_session_tokens = max_session_tokens
        self._input_tokens_used: int = 0
        self._output_tokens_used: int = 0

    @property
    def max_session_tokens(self) -> int:
        """Return the maximum session token budget."""
        return self._max_session_tokens

    @property
    def total_tokens_used(self) -> int:
        """Return total tokens used (input + output)."""
        return self._input_tokens_used + self._output_tokens_used

    @property
    def input_tokens_used(self) -> int:
        """Return input tokens used."""
        return self._input_tokens_used

    @property
    def output_tokens_used(self) -> int:
        """Return output tokens used."""
        return self._output_tokens_used

    @property
    def remaining(self) -> int:
        """Return remaining token budget."""
        return max(0, self._max_session_tokens - self.total_tokens_used)

    def pre_check(self, estimated_tokens: int = 0) -> None:
        """Check if the budget allows another LLM call.

        Args:
            estimated_tokens: Estimated tokens for the next call.

        Raises:
            LLMBudgetExceededError: If budget is exhausted or would be exceeded.
        """
        if self.total_tokens_used >= self._max_session_tokens:
            raise LLMBudgetExceededError(
                f"Session token budget exhausted: "
                f"{self.total_tokens_used}/{self._max_session_tokens} tokens used",
                details={
                    "total_used": self.total_tokens_used,
                    "max_allowed": self._max_session_tokens,
                },
            )

        if (
            estimated_tokens > 0
            and self.total_tokens_used + estimated_tokens > self._max_session_tokens
        ):
            raise LLMBudgetExceededError(
                f"Estimated tokens ({estimated_tokens}) would exceed budget: "
                f"{self.total_tokens_used + estimated_tokens}/{self._max_session_tokens}",
                details={
                    "total_used": self.total_tokens_used,
                    "estimated": estimated_tokens,
                    "max_allowed": self._max_session_tokens,
                },
            )

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record actual token usage from an LLM response.

        Args:
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens consumed.
        """
        self._input_tokens_used += input_tokens
        self._output_tokens_used += output_tokens
