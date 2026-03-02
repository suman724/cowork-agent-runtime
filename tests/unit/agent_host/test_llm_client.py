"""Tests for LLMClient — streaming, retry, timeout, text chunking, tool call parsing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

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

    async def test_raises_after_max_retries(self, client: LLMClient) -> None:
        """Should raise after exhausting retries."""

        async def mock_do_stream(messages, tools, on_text_chunk):
            raise httpx.ConnectError("connection failed")

        client._do_stream = mock_do_stream  # type: ignore[assignment]
        with pytest.raises(httpx.ConnectError):
            await client.stream_chat([{"role": "user", "content": "hi"}])

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
