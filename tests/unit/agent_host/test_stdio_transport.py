"""Tests for StdioTransport — async stdin/stdout I/O."""

from __future__ import annotations

import asyncio
from io import StringIO

import pytest

from agent_host.server.stdio_transport import StdioTransport


class TestStdioTransport:
    @pytest.mark.asyncio
    async def test_read_message(self) -> None:
        """read_message returns a line from stdin."""
        reader = asyncio.StreamReader()
        reader.feed_data(b'{"method":"test"}\n')
        reader.feed_eof()

        transport = StdioTransport(reader=reader)
        msg = await transport.read_message()
        assert msg == '{"method":"test"}'

    @pytest.mark.asyncio
    async def test_read_message_eof(self) -> None:
        """read_message returns None on EOF."""
        reader = asyncio.StreamReader()
        reader.feed_eof()

        transport = StdioTransport(reader=reader)
        msg = await transport.read_message()
        assert msg is None

    @pytest.mark.asyncio
    async def test_read_message_no_reader(self) -> None:
        """read_message returns None when no reader configured."""
        transport = StdioTransport()
        msg = await transport.read_message()
        assert msg is None

    @pytest.mark.asyncio
    async def test_write_message(self) -> None:
        """write_message writes to stdout with newline."""
        output = StringIO()
        transport = StdioTransport(writer=output)

        await transport.write_message('{"result":"ok"}')
        assert output.getvalue() == '{"result":"ok"}\n'

    @pytest.mark.asyncio
    async def test_write_message_concurrent(self) -> None:
        """Concurrent writes don't interleave (lock ensures serialization)."""
        output = StringIO()
        transport = StdioTransport(writer=output)

        await asyncio.gather(
            transport.write_message("msg1"),
            transport.write_message("msg2"),
        )

        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 2
        assert set(lines) == {"msg1", "msg2"}

    def test_write_sync(self) -> None:
        """write_sync writes synchronously."""
        output = StringIO()
        transport = StdioTransport(writer=output)

        transport.write_sync('{"notify":"event"}')
        assert output.getvalue() == '{"notify":"event"}\n'

    @pytest.mark.asyncio
    async def test_read_multiple_messages(self) -> None:
        """read_message reads one line at a time."""
        reader = asyncio.StreamReader()
        reader.feed_data(b'{"id":1}\n{"id":2}\n')
        reader.feed_eof()

        transport = StdioTransport(reader=reader)
        msg1 = await transport.read_message()
        msg2 = await transport.read_message()
        msg3 = await transport.read_message()

        assert msg1 == '{"id":1}'
        assert msg2 == '{"id":2}'
        assert msg3 is None
