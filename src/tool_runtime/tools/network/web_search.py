"""WebSearch tool — web search via Tavily API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import httpx

from tool_runtime.exceptions import ToolExecutionError, ToolTimeoutError
from tool_runtime.output.artifacts import maybe_extract_artifact
from tool_runtime.tools.base import BaseTool

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

_TAVILY_API_URL = "https://api.tavily.com/search"
_DEFAULT_MAX_RESULTS = 5
_ABSOLUTE_MAX_RESULTS = 10
_SEARCH_TIMEOUT_SECONDS = 30


class WebSearchTool(BaseTool):
    """Search the web using Tavily API."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client

    @property
    def name(self) -> str:
        return "WebSearch"

    @property
    def description(self) -> str:
        return (
            "Search the web for information using the Tavily search API. "
            "Returns titles, URLs, and content snippets for matching results."
        )

    @property
    def capability(self) -> str:
        return "Search.Web"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "minimum": 1,
                    "maximum": _ABSOLUTE_MAX_RESULTS,
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ExecutionContext,  # noqa: ARG002
    ) -> RawToolOutput:
        self.validate_input(arguments)
        query: str = arguments["query"]
        max_results: int = min(
            arguments.get("max_results", _DEFAULT_MAX_RESULTS),
            _ABSOLUTE_MAX_RESULTS,
        )

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ToolExecutionError(
                "TAVILY_API_KEY not configured. Set the TAVILY_API_KEY environment variable."
            )

        client = self._client or httpx.AsyncClient()
        should_close = self._client is None

        try:
            response = await client.post(
                _TAVILY_API_URL,
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=_SEARCH_TIMEOUT_SECONDS,
            )

            if response.status_code != 200:
                raise ToolExecutionError(
                    f"Tavily API returned status {response.status_code}: {response.text}"
                )

            try:
                data = response.json()
            except Exception as e:
                raise ToolExecutionError(f"Failed to parse Tavily response: {e}") from e

            results = data.get("results", [])
            output_text = _format_results(query, results)

        except httpx.TimeoutException as e:
            raise ToolTimeoutError(f"Search request timed out: {e}") from e
        except httpx.HTTPError as e:
            raise ToolExecutionError(f"Search request failed: {e}") from e
        finally:
            if should_close:
                await client.aclose()

        return maybe_extract_artifact(output_text, "search_results", "search.txt")


def _format_results(query: str, results: list[dict[str, Any]]) -> str:
    """Format search results as a numbered list."""
    if not results:
        return f"No results found for: {query}"

    lines: list[str] = [f"Search results for: {query}\n"]
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        lines.append(f"{i}. **{title}** ({url})")
        if content:
            # Truncate long snippets
            snippet = content[:300].strip()
            if len(content) > 300:
                snippet += "..."
            lines.append(f"   {snippet}")
        lines.append("")

    return "\n".join(lines)
