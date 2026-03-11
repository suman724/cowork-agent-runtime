"""Async stdin/stdout transport for JSON-RPC over stdio.

One JSON message per line. stdout has a write lock to prevent
interleaving of responses and notifications.

Implements the ``Transport`` protocol.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, TextIO

import structlog

from agent_host.server.json_rpc import serialize_notification

logger = structlog.get_logger()


class StdioTransport:
    """Newline-delimited JSON transport over stdin/stdout.

    - read_message(): reads one line from stdin (async)
    - write_message(): writes one line to stdout with lock (async)
    - send_event(): sends a JSON-RPC notification (sync, fire-and-forget)
    - start() / shutdown(): no-ops for stdio (satisfies Transport protocol)
    """

    def __init__(
        self,
        reader: asyncio.StreamReader | None = None,
        writer: TextIO | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer or sys.stdout
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        """No-op for stdio transport."""

    async def read_message(self) -> str | None:
        """Read one JSON message (one line) from stdin.

        Returns None on EOF.
        """
        if self._reader is None:
            return None

        try:
            line = await self._reader.readline()
            if not line:
                return None
            return line.decode("utf-8").strip()
        except (asyncio.CancelledError, ConnectionResetError):
            return None

    async def write_message(self, message: str) -> None:
        """Write one JSON message to stdout (async, with lock)."""
        async with self._write_lock:
            self._writer.write(message + "\n")
            self._writer.flush()

    def send_event(self, event: dict[str, Any]) -> None:
        """Send a JSON-RPC notification to stdout (fire-and-forget).

        Replaces the old ``write_sync()`` method.
        """
        try:
            notification = serialize_notification("SessionEvent", event)
            self._writer.write(notification + "\n")
            self._writer.flush()
        except Exception:
            logger.warning("event_notification_failed", exc_info=True)

    async def shutdown(self) -> None:
        """No-op for stdio transport."""
