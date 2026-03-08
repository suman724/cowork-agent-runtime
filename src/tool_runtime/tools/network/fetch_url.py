"""FetchUrl tool — fetch a URL and extract readable text content."""

from __future__ import annotations

import json
from typing import Any

import httpx

from tool_runtime.exceptions import ToolExecutionError, ToolTimeoutError
from tool_runtime.models import (
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_MAX_RESPONSE_BYTES,
    ExecutionContext,
    RawToolOutput,
)
from tool_runtime.output.artifacts import maybe_extract_artifact
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import validate_url


class FetchUrlTool(BaseTool):
    """Fetch a URL and extract readable text content."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client

    @property
    def name(self) -> str:
        return "FetchUrl"

    @property
    def description(self) -> str:
        return (
            "Fetch a URL and extract readable text content. "
            "HTML pages are converted to markdown for readability. "
            "JSON responses are pretty-printed. Plain text is returned as-is."
        )

    @property
    def capability(self) -> str:
        return "Network.Http"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Request timeout in seconds.",
                    "minimum": 1,
                    "maximum": 120,
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> RawToolOutput:
        self.validate_input(arguments)
        url: str = arguments["url"]
        timeout: int = arguments.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS)

        validate_url(url)

        client = self._client or httpx.AsyncClient()
        should_close = self._client is None

        try:
            response = await client.get(
                url,
                timeout=timeout,
                follow_redirects=True,
            )

            response_text = response.text
            response_size = len(response_text.encode("utf-8"))
            if response_size > DEFAULT_MAX_RESPONSE_BYTES:
                response_text = response_text[: DEFAULT_MAX_RESPONSE_BYTES // 4]
                response_text += "\n\n[Content truncated — response too large]"

            content_type = response.headers.get("content-type", "")
            output_text = _format_response(response_text, content_type, url)

        except httpx.TimeoutException as e:
            raise ToolTimeoutError(f"Request timed out after {timeout}s: {e}") from e
        except httpx.HTTPError as e:
            raise ToolExecutionError(f"HTTP request failed: {e}") from e
        finally:
            if should_close:
                await client.aclose()

        max_output = context.max_output_bytes
        return maybe_extract_artifact(output_text, "tool_output", "page.md", max_output)


def _format_response(text: str, content_type: str, url: str) -> str:
    """Format the response based on content type."""
    ct_lower = content_type.lower()

    if "text/html" in ct_lower or "application/xhtml" in ct_lower:
        return _html_to_markdown(text, url)

    if "application/json" in ct_lower or "text/json" in ct_lower:
        return _format_json(text, url)

    return f"Fetched: {url}\n\n{text}"


def _html_to_markdown(html: str, url: str) -> str:
    """Convert HTML to markdown for readability."""
    try:
        import markdownify

        md = markdownify.markdownify(html, strip=["script", "style", "nav", "footer"])
        # Clean up excessive whitespace from conversion
        lines = md.splitlines()
        cleaned: list[str] = []
        blank_count = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append("")
            else:
                blank_count = 0
                cleaned.append(line)
        md = "\n".join(cleaned).strip()
        return f"Fetched: {url}\n\n{md}"
    except ImportError:
        # markdownify not installed — return raw text with tags stripped
        import re

        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return f"Fetched: {url}\n\n{text}"


def _format_json(text: str, url: str) -> str:
    """Pretty-print JSON response."""
    try:
        data = json.loads(text)
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        return f"Fetched: {url}\n\n{formatted}"
    except (json.JSONDecodeError, ValueError):
        return f"Fetched: {url}\n\n{text}"
