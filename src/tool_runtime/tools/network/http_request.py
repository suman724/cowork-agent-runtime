"""HttpRequest tool — HTTP requests with SSRF prevention."""

from __future__ import annotations

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


class HttpRequestTool(BaseTool):
    """Make HTTP requests."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client

    @property
    def name(self) -> str:
        return "HttpRequest"

    @property
    def description(self) -> str:
        return (
            "Make an HTTP request to a URL. Supports GET, POST, PUT, PATCH, DELETE. "
            "Includes SSRF prevention — requests to private/loopback IPs are blocked."
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
                    "description": "The URL to send the request to.",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method.",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    "default": "GET",
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers.",
                    "additionalProperties": {"type": "string"},
                },
                "body": {
                    "type": "string",
                    "description": "Request body (for POST/PUT/PATCH).",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Request timeout in seconds.",
                    "minimum": 1,
                    "default": DEFAULT_HTTP_TIMEOUT_SECONDS,
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
        method: str = arguments.get("method", "GET")
        headers: dict[str, str] = arguments.get("headers", {})
        body: str | None = arguments.get("body")
        timeout: int = arguments.get("timeout_seconds", DEFAULT_HTTP_TIMEOUT_SECONDS)

        validate_url(url)

        client = self._client or httpx.AsyncClient()
        should_close = self._client is None

        try:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                content=body.encode("utf-8") if body else None,
                timeout=timeout,
                follow_redirects=True,
            )

            # Check response size via Content-Length header (best-effort early check)
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > DEFAULT_MAX_RESPONSE_BYTES:
                        raise ToolExecutionError(
                            f"Response too large: {content_length} bytes "
                            f"(limit: {DEFAULT_MAX_RESPONSE_BYTES})"
                        )
                except ValueError:
                    pass  # Malformed Content-Length — fall through to actual size check

            response_text = response.text
            response_size = len(response_text.encode("utf-8"))
            if response_size > DEFAULT_MAX_RESPONSE_BYTES:
                raise ToolExecutionError(
                    f"Response too large: {response_size} bytes "
                    f"(limit: {DEFAULT_MAX_RESPONSE_BYTES})"
                )

            content_type = response.headers.get("content-type", "")
            output_text = (
                f"HTTP {response.status_code} {response.reason_phrase}\n"
                f"Content-Type: {content_type}\n\n"
                f"{response_text}"
            )
        except httpx.TimeoutException as e:
            raise ToolTimeoutError(f"HTTP request timed out after {timeout}s: {e}") from e
        except httpx.HTTPError as e:
            raise ToolExecutionError(f"HTTP request failed: {e}") from e
        finally:
            if should_close:
                await client.aclose()

        max_output = context.max_output_bytes
        return maybe_extract_artifact(output_text, "http_response", "response.txt", max_output)
