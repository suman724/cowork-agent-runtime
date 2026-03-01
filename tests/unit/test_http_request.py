"""Tests for HttpRequest tool."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from tool_runtime.exceptions import ToolExecutionError, ToolInputValidationError, ToolTimeoutError
from tool_runtime.models import ExecutionContext
from tool_runtime.tools.network.http_request import HttpRequestTool


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


def _make_mock_transport(
    status_code: int = 200,
    body: str = "OK",
    content_type: str = "text/plain",
    headers: dict[str, str] | None = None,
) -> httpx.MockTransport:
    """Create an httpx mock transport that returns a fixed response."""
    resp_headers = {"content-type": content_type}
    if headers:
        resp_headers.update(headers)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            text=body,
            headers=resp_headers,
        )

    return httpx.MockTransport(handler)


class TestHttpRequest:
    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_get_request(self, _mock: object, context: ExecutionContext) -> None:
        transport = _make_mock_transport(200, '{"key": "value"}', "application/json")
        client = httpx.AsyncClient(transport=transport)
        tool = HttpRequestTool(http_client=client)

        result = await tool.execute({"url": "https://api.example.com/data"}, context)
        assert "HTTP 200" in result.output_text
        assert '{"key": "value"}' in result.output_text
        await client.aclose()

    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_post_request(self, _mock: object, context: ExecutionContext) -> None:
        transport = _make_mock_transport(201, "Created")
        client = httpx.AsyncClient(transport=transport)
        tool = HttpRequestTool(http_client=client)

        result = await tool.execute(
            {"url": "https://api.example.com/data", "method": "POST", "body": '{"x": 1}'},
            context,
        )
        assert "HTTP 201" in result.output_text
        await client.aclose()

    async def test_ssrf_rejected(self, context: ExecutionContext) -> None:
        tool = HttpRequestTool()
        with pytest.raises(ToolInputValidationError, match="private"):
            await tool.execute({"url": "http://127.0.0.1/admin"}, context)

    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_timeout(self, _mock: object, context: ExecutionContext) -> None:
        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        transport = httpx.MockTransport(timeout_handler)
        client = httpx.AsyncClient(transport=transport)
        tool = HttpRequestTool(http_client=client)

        with pytest.raises(ToolTimeoutError, match="timed out"):
            await tool.execute(
                {"url": "https://api.example.com/slow", "timeout_seconds": 1},
                context,
            )
        await client.aclose()

    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_large_response_rejected(self, _mock: object, context: ExecutionContext) -> None:
        transport = _make_mock_transport(
            200,
            "x" * (10 * 1024 * 1024 + 1),  # Just over 10MB
        )
        client = httpx.AsyncClient(transport=transport)
        tool = HttpRequestTool(http_client=client)

        with pytest.raises(ToolExecutionError, match="too large"):
            await tool.execute({"url": "https://api.example.com/huge"}, context)
        await client.aclose()

    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_malformed_content_length_ignored(
        self, _mock: object, context: ExecutionContext
    ) -> None:
        """Malformed Content-Length header should not crash — falls through to body check."""
        transport = _make_mock_transport(200, "OK", headers={"content-length": "not-a-number"})
        client = httpx.AsyncClient(transport=transport)
        tool = HttpRequestTool(http_client=client)

        result = await tool.execute({"url": "https://api.example.com/data"}, context)
        assert "HTTP 200" in result.output_text
        await client.aclose()
