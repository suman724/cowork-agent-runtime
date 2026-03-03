"""Tests for LLMClient — streaming, retry, timeout, text chunking, tool call parsing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent_host.exceptions import LLMGatewayError
from agent_host.llm.client import LLMClient
from agent_host.llm.models import LLMResponse


class TestLLMClientRetry:
    """Test retry behavior on transient errors."""

    @pytest.fixture
    def client(self) -> LLMClient:
        return LLMClient(
            endpoint="https://llm.example.com",
            auth_token="test-token",
            model="gpt-4o",
            max_retries=2,
            retry_base_delay=0.01,
            retry_max_delay=0.05,
        )

    async def test_retries_on_transient_error(self, client: LLMClient) -> None:
        """Should retry on transient errors up to max_retries."""
        call_count = 0

        async def mock_do_stream(messages, tools, on_text_chunk):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("connection failed")
            return LLMResponse(text="OK", stop_reason="stop", input_tokens=5, output_tokens=3)

        client._do_stream = mock_do_stream  # type: ignore[assignment]
        result = await client.stream_chat([{"role": "user", "content": "hi"}])
        assert result.text == "OK"
        assert call_count == 3

    async def test_raises_on_permanent_error(self, client: LLMClient) -> None:
        """Should not retry permanent errors."""
        from agent_host.exceptions import LLMBudgetExceededError

        async def mock_do_stream(messages, tools, on_text_chunk):
            raise LLMBudgetExceededError("budget exhausted")

        client._do_stream = mock_do_stream  # type: ignore[assignment]
        with pytest.raises(LLMBudgetExceededError):
            await client.stream_chat([{"role": "user", "content": "hi"}])

    async def test_wraps_in_llm_gateway_error_after_max_retries(self, client: LLMClient) -> None:
        """Should wrap transient errors in LLMGatewayError after exhausting retries."""

        async def mock_do_stream(messages, tools, on_text_chunk):
            raise httpx.ConnectError("connection failed")

        client._do_stream = mock_do_stream  # type: ignore[assignment]
        with pytest.raises(LLMGatewayError, match="temporarily unavailable"):
            await client.stream_chat([{"role": "user", "content": "hi"}])

    async def test_wraps_rate_limit_with_friendly_message(self, client: LLMClient) -> None:
        """Should wrap rate limit errors with a rate-limit-specific message."""

        class RateLimitError(Exception):
            status_code = 429

        async def mock_do_stream(messages, tools, on_text_chunk):
            raise RateLimitError("429 Too Many Requests")

        client._do_stream = mock_do_stream  # type: ignore[assignment]
        with pytest.raises(LLMGatewayError, match="Rate limited") as exc_info:
            await client.stream_chat([{"role": "user", "content": "hi"}])
        assert exc_info.value.details.get("error_type") == "rate_limit"

    async def test_uses_retry_after_header(self, client: LLMClient) -> None:
        """Should respect Retry-After header from rate limit responses."""
        call_count = 0

        class RateLimitError(Exception):
            status_code = 429

            def __init__(self) -> None:
                super().__init__("429")
                self.response = MagicMock()
                self.response.headers = {"retry-after": "0.01"}

        async def mock_do_stream(messages, tools, on_text_chunk):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError()
            return LLMResponse(text="OK", stop_reason="stop")

        client._do_stream = mock_do_stream  # type: ignore[assignment]
        result = await client.stream_chat([{"role": "user", "content": "hi"}])
        assert result.text == "OK"
        assert call_count == 2

    async def test_529_uses_longer_backoff(self, client: LLMClient) -> None:
        """Should use 3x base delay for 529 Overloaded errors."""
        import asyncio
        from unittest.mock import patch

        delays: list[float] = []
        call_count = 0

        class OverloadedError(Exception):
            status_code = 529

        async def mock_do_stream(messages, tools, on_text_chunk):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OverloadedError("529 Overloaded")
            return LLMResponse(text="OK", stop_reason="stop", input_tokens=5, output_tokens=3)

        client._do_stream = mock_do_stream  # type: ignore[assignment]

        original_sleep = asyncio.sleep

        async def capture_sleep(delay: float) -> None:
            delays.append(delay)
            await original_sleep(0)  # Don't actually wait

        with patch("asyncio.sleep", side_effect=capture_sleep):
            result = await client.stream_chat([{"role": "user", "content": "hi"}])

        assert result.text == "OK"
        assert call_count == 3
        assert len(delays) == 2

        # base=0.01, 529 uses 3x: attempt 0 → 0.01*3*1=0.03, attempt 1 → 0.01*3*2=0.06
        # Plus up to 25% jitter, so check the delay is >= base*3*2^attempt
        assert delays[0] >= 0.03  # 0.01 * 3 * 2^0
        assert delays[1] >= 0.06  # 0.01 * 3 * 2^1

    async def test_non_529_transient_uses_standard_backoff(self, client: LLMClient) -> None:
        """Non-529 transient errors should use standard 1x base delay."""
        import asyncio
        from unittest.mock import patch

        delays: list[float] = []
        call_count = 0

        async def mock_do_stream(messages, tools, on_text_chunk):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("connection failed")
            return LLMResponse(text="OK", stop_reason="stop", input_tokens=5, output_tokens=3)

        client._do_stream = mock_do_stream  # type: ignore[assignment]

        original_sleep = asyncio.sleep

        async def capture_sleep(delay: float) -> None:
            delays.append(delay)
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            result = await client.stream_chat([{"role": "user", "content": "hi"}])

        assert result.text == "OK"
        assert len(delays) == 2

        # base=0.01, standard: attempt 0 → 0.01*1=0.01, attempt 1 → 0.01*2=0.02
        # Plus up to 25% jitter
        assert delays[0] >= 0.01
        assert delays[0] < 0.03  # Must be less than 529's first delay
        assert delays[1] >= 0.02
        assert delays[1] < 0.06  # Must be less than 529's second delay

    async def test_emits_retry_event(self, client: LLMClient) -> None:
        """Should emit llm_retry event on transient error."""
        call_count = 0
        event_emitter = MagicMock()
        client._event_emitter = event_emitter

        async def mock_do_stream(messages, tools, on_text_chunk):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("connection failed")
            return LLMResponse(text="OK", stop_reason="stop")

        client._do_stream = mock_do_stream  # type: ignore[assignment]
        await client.stream_chat([{"role": "user", "content": "hi"}], task_id="t1")
        event_emitter.emit_llm_retry.assert_called_once()


class TestLLMClientClose:
    async def test_close(self) -> None:
        client = LLMClient(
            endpoint="https://llm.example.com",
            auth_token="test-token",
            model="gpt-4o",
        )
        client._client = AsyncMock()
        await client.close()
        client._client.close.assert_awaited_once()
