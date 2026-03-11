"""Transport protocol — abstract interface for agent host communication."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """Protocol for agent host transport implementations.

    Two implementations:
    - StdioTransport: JSON-RPC over stdin/stdout (desktop app, default)
    - HttpTransport: HTTP/SSE server (web/sandbox mode)
    """

    async def start(self) -> None:
        """Start the transport (e.g., bind HTTP server port).

        For StdioTransport this is a no-op. For HttpTransport this starts
        the ASGI server.
        """
        ...

    def send_event(self, event: dict[str, Any]) -> None:
        """Send an event notification to the client.

        Fire-and-forget — errors are logged but never propagated.
        For StdioTransport this writes a JSON-RPC notification to stdout.
        For HttpTransport this pushes to the event buffer for SSE streaming.
        """
        ...

    async def shutdown(self) -> None:
        """Gracefully shut down the transport.

        For StdioTransport this is a no-op. For HttpTransport this stops
        the ASGI server and closes SSE connections.
        """
        ...
