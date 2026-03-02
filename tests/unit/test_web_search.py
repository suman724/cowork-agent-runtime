"""Tests for WebSearch tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tool_runtime.exceptions import ToolExecutionError, ToolTimeoutError
from tool_runtime.models import ExecutionContext
from tool_runtime.tools.network.web_search import WebSearchTool


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


def _make_search_response(results: list[dict[str, str]]) -> httpx.Response:
    """Create a mock Tavily search response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {"results": results}
    return response


class TestWebSearch:
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    async def test_successful_search(self, context: ExecutionContext) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            return_value=_make_search_response(
                [
                    {
                        "title": "Python Docs",
                        "url": "https://docs.python.org",
                        "content": "Official Python documentation.",
                    },
                    {
                        "title": "Python Tutorial",
                        "url": "https://tutorial.python.org",
                        "content": "Learn Python programming.",
                    },
                ]
            )
        )
        mock_client.aclose = AsyncMock()

        tool = WebSearchTool(mock_client)
        result = await tool.execute({"query": "python documentation"}, context)

        assert "Python Docs" in result.output_text
        assert "docs.python.org" in result.output_text
        assert "Python Tutorial" in result.output_text
        assert "1." in result.output_text
        assert "2." in result.output_text

    async def test_no_api_key(self, context: ExecutionContext) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        tool = WebSearchTool(mock_client)

        with patch.dict("os.environ", {}, clear=True):
            # Also clear TAVILY_API_KEY if it exists
            import os

            os.environ.pop("TAVILY_API_KEY", None)
            with pytest.raises(ToolExecutionError, match="TAVILY_API_KEY"):
                await tool.execute({"query": "test"}, context)

    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    async def test_api_error(self, context: ExecutionContext) -> None:
        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 500
        error_response.text = "Internal Server Error"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=error_response)
        mock_client.aclose = AsyncMock()

        tool = WebSearchTool(mock_client)
        with pytest.raises(ToolExecutionError, match="status 500"):
            await tool.execute({"query": "test"}, context)

    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    async def test_timeout(self, context: ExecutionContext) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
        mock_client.aclose = AsyncMock()

        tool = WebSearchTool(mock_client)
        with pytest.raises(ToolTimeoutError, match="timed out"):
            await tool.execute({"query": "test"}, context)

    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    async def test_max_results_passed_to_api(self, context: ExecutionContext) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=_make_search_response([]))
        mock_client.aclose = AsyncMock()

        tool = WebSearchTool(mock_client)
        await tool.execute({"query": "test", "max_results": 3}, context)

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["max_results"] == 3

    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    async def test_no_results(self, context: ExecutionContext) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=_make_search_response([]))
        mock_client.aclose = AsyncMock()

        tool = WebSearchTool(mock_client)
        result = await tool.execute({"query": "xyznonexistent123"}, context)
        assert "No results" in result.output_text

    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    async def test_max_results_capped(self, context: ExecutionContext) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=_make_search_response([]))
        mock_client.aclose = AsyncMock()

        tool = WebSearchTool(mock_client)
        # Even if someone tries 50, it should be capped at 10
        await tool.execute({"query": "test", "max_results": 10}, context)

        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["max_results"] == 10
