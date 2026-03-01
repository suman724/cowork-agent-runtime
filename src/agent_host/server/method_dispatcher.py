"""Method dispatcher — routes JSON-RPC method names to handler functions."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import structlog

from agent_host.exceptions import AgentHostError
from agent_host.server.json_rpc import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    JsonRpcRequest,
    JsonRpcResponse,
    make_error_response,
    make_success_response,
)

logger = structlog.get_logger()

# Handler type: async function taking params dict, returning result dict
Handler = Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


class MethodDispatcher:
    """Routes JSON-RPC method names to async handler functions.

    Catches all exceptions and maps them to JSON-RPC error responses.
    AgentHostError subclasses use their json_rpc_code; other exceptions
    become internal errors.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, method: str, handler: Handler) -> None:
        """Register a handler for a JSON-RPC method name."""
        self._handlers[method] = handler

    async def dispatch(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Dispatch a request to the registered handler.

        Returns a JsonRpcResponse (success or error). Never raises.
        """
        handler = self._handlers.get(request.method)
        if handler is None:
            return make_error_response(
                request.id,
                METHOD_NOT_FOUND,
                f"Method not found: {request.method}",
            )

        try:
            result = await handler(request.params)
            return make_success_response(request.id, result)
        except AgentHostError as e:
            logger.warning(
                "handler_error",
                method=request.method,
                error_type=type(e).__name__,
                message=e.message,
            )
            return make_error_response(
                request.id,
                e.json_rpc_code,
                e.message,
                data=e.details if e.details else None,
            )
        except Exception:
            logger.exception("handler_internal_error", method=request.method)
            return make_error_response(
                request.id,
                INTERNAL_ERROR,
                "Internal error",
            )
