"""JSON-RPC 2.0 protocol types, parsing, and serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


@dataclass
class JsonRpcError(Exception):
    """JSON-RPC 2.0 error object.

    Extends Exception so it can be raised by parse_request()
    and caught with pytest.raises() or try/except.
    """

    code: int
    message: str
    data: dict[str, Any] | None = None


@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 request object."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | int | None = None

    @property
    def is_notification(self) -> bool:
        """Notifications have no id — no response expected."""
        return self.id is None


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 response object."""

    id: str | int | None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


def parse_request(raw: str) -> JsonRpcRequest:
    """Parse a JSON-RPC 2.0 request from a raw JSON string.

    Raises:
        JsonRpcError: If the request is malformed.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise JsonRpcError(PARSE_ERROR, f"Parse error: {e}") from e

    if not isinstance(data, dict):
        raise JsonRpcError(INVALID_REQUEST, "Request must be a JSON object")

    method = data.get("method")
    if not isinstance(method, str):
        raise JsonRpcError(INVALID_REQUEST, "Missing or invalid 'method' field")

    params = data.get("params", {})
    if not isinstance(params, dict):
        raise JsonRpcError(INVALID_PARAMS, "'params' must be a JSON object")

    return JsonRpcRequest(
        method=method,
        params=params,
        id=data.get("id"),
    )


def serialize_response(response: JsonRpcResponse) -> str:
    """Serialize a JSON-RPC 2.0 response to a JSON string."""
    result: dict[str, Any] = {"jsonrpc": "2.0", "id": response.id}

    if response.error is not None:
        error_obj: dict[str, Any] = {
            "code": response.error.code,
            "message": response.error.message,
        }
        if response.error.data is not None:
            error_obj["data"] = response.error.data
        result["error"] = error_obj
    else:
        result["result"] = response.result or {}

    return json.dumps(result, separators=(",", ":"))


def serialize_notification(method: str, params: dict[str, Any]) -> str:
    """Serialize a JSON-RPC 2.0 notification (no id, no response expected)."""
    notification: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    }
    return json.dumps(notification, separators=(",", ":"))


def make_error_response(
    request_id: str | int | None,
    code: int,
    message: str,
    data: dict[str, Any] | None = None,
) -> JsonRpcResponse:
    """Create a JSON-RPC error response."""
    return JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(code=code, message=message, data=data),
    )


def make_success_response(
    request_id: str | int | None,
    result: dict[str, Any],
) -> JsonRpcResponse:
    """Create a JSON-RPC success response."""
    return JsonRpcResponse(id=request_id, result=result)
