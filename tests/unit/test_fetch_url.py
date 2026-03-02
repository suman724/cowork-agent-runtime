"""Tests for FetchUrl tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tool_runtime.exceptions import ToolExecutionError, ToolInputValidationError, ToolTimeoutError
from tool_runtime.models import ExecutionContext
from tool_runtime.tools.network.fetch_url import FetchUrlTool


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


def _make_response(
    text: str, content_type: str = "text/html", status_code: int = 200
) -> httpx.Response:
    """Create a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.text = text
    response.status_code = status_code
    response.headers = {"content-type": content_type}
    return response


class TestFetchUrl:
    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_html_to_markdown(self, _mock_ip: MagicMock, context: ExecutionContext) -> None:
        html = "<html><body><h1>Title</h1><p>Paragraph text.</p></body></html>"
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_make_response(html, "text/html"))
        mock_client.aclose = AsyncMock()

        tool = FetchUrlTool(mock_client)
        result = await tool.execute({"url": "https://example.com/page"}, context)

        assert "example.com" in result.output_text
        # Content should be present (either markdownified or tag-stripped)
        assert "Title" in result.output_text
        assert "Paragraph" in result.output_text

    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_json_response(self, _mock_ip: MagicMock, context: ExecutionContext) -> None:
        json_text = '{"key": "value", "num": 42}'
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_make_response(json_text, "application/json"))
        mock_client.aclose = AsyncMock()

        tool = FetchUrlTool(mock_client)
        result = await tool.execute({"url": "https://api.example.com/data"}, context)

        assert '"key": "value"' in result.output_text
        assert '"num": 42' in result.output_text

    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_plain_text(self, _mock_ip: MagicMock, context: ExecutionContext) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_make_response("Hello, world!", "text/plain"))
        mock_client.aclose = AsyncMock()

        tool = FetchUrlTool(mock_client)
        result = await tool.execute({"url": "https://example.com/text"}, context)

        assert "Hello, world!" in result.output_text

    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_timeout(self, _mock_ip: MagicMock, context: ExecutionContext) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
        mock_client.aclose = AsyncMock()

        tool = FetchUrlTool(mock_client)
        with pytest.raises(ToolTimeoutError, match="timed out"):
            await tool.execute({"url": "https://slow.example.com"}, context)

    async def test_ssrf_blocked(self, context: ExecutionContext) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        tool = FetchUrlTool(mock_client)

        with pytest.raises(ToolInputValidationError, match="private"):
            await tool.execute({"url": "http://127.0.0.1/admin"}, context)

    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_http_error(self, _mock_ip: MagicMock, context: ExecutionContext) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client.aclose = AsyncMock()

        tool = FetchUrlTool(mock_client)
        with pytest.raises(ToolExecutionError, match="HTTP request failed"):
            await tool.execute({"url": "https://unreachable.example.com"}, context)

    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    async def test_large_response_truncated(
        self, _mock_ip: MagicMock, context: ExecutionContext
    ) -> None:
        # 20MB response
        large_text = "x" * (20 * 1024 * 1024)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_make_response(large_text, "text/plain"))
        mock_client.aclose = AsyncMock()

        tool = FetchUrlTool(mock_client)
        result = await tool.execute({"url": "https://example.com/large"}, context)

        assert "truncated" in result.output_text.lower()
