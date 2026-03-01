"""Async stdin/stdout transport for JSON-RPC over stdio.

One JSON message per line. stdout has a write lock to prevent
interleaving of responses and notifications.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TextIO


class StdioTransport:
    """Newline-delimited JSON transport over stdin/stdout.

    - read_message(): reads one line from stdin (async)
    - write_message(): writes one line to stdout with lock (async)
    - write_sync(): writes one line to stdout synchronously (for fire-and-forget)
    """

    def __init__(
        self,
        reader: asyncio.StreamReader | None = None,
        writer: TextIO | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer or sys.stdout
        self._write_lock = asyncio.Lock()

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

    def write_sync(self, message: str) -> None:
        """Write one JSON message to stdout synchronously.

        Used for fire-and-forget notifications from non-async contexts.
        """
        self._writer.write(message + "\n")
        self._writer.flush()
