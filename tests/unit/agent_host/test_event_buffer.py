"""Tests for EventBuffer — bounded ring buffer for SSE event replay."""

from __future__ import annotations

import asyncio

import pytest

from agent_host.server.event_buffer import EventBuffer


class TestEventBuffer:
    def test_push_assigns_monotonic_ids(self) -> None:
        """Events get sequential IDs starting from 1."""
        buf = EventBuffer(capacity=100)
        id1 = buf.push({"type": "a"})
        id2 = buf.push({"type": "b"})
        id3 = buf.push({"type": "c"})
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_size_and_min_max_id(self) -> None:
        """Properties reflect current buffer state."""
        buf = EventBuffer(capacity=100)
        assert buf.size == 0
        assert buf.min_id == 0
        assert buf.max_id == 0

        buf.push({"type": "a"})
        buf.push({"type": "b"})
        assert buf.size == 2
        assert buf.min_id == 1
        assert buf.max_id == 2

    def test_capacity_eviction(self) -> None:
        """Oldest events are evicted when capacity is exceeded."""
        buf = EventBuffer(capacity=3)
        buf.push({"n": 1})
        buf.push({"n": 2})
        buf.push({"n": 3})
        buf.push({"n": 4})  # evicts event 1

        assert buf.size == 3
        assert buf.min_id == 2
        assert buf.max_id == 4

    def test_get_since_zero_returns_all(self) -> None:
        """since=0 returns all buffered events."""
        buf = EventBuffer(capacity=100)
        buf.push({"n": 1})
        buf.push({"n": 2})

        events, gap = buf.get_since(0)
        assert len(events) == 2
        assert events[0].id == 1
        assert events[1].id == 2
        assert gap is False

    def test_get_since_returns_after_id(self) -> None:
        """Returns only events with id > since_id."""
        buf = EventBuffer(capacity=100)
        buf.push({"n": 1})
        buf.push({"n": 2})
        buf.push({"n": 3})

        events, gap = buf.get_since(1)
        assert len(events) == 2
        assert events[0].id == 2
        assert events[1].id == 3
        assert gap is False

    def test_get_since_beyond_max_returns_empty(self) -> None:
        """since > max_id returns empty list."""
        buf = EventBuffer(capacity=100)
        buf.push({"n": 1})
        buf.push({"n": 2})

        events, gap = buf.get_since(5)
        assert len(events) == 0
        assert gap is False

    def test_get_since_detects_gap(self) -> None:
        """Gap detected when since_id < min_id - 1 (events were evicted)."""
        buf = EventBuffer(capacity=3)
        buf.push({"n": 1})
        buf.push({"n": 2})
        buf.push({"n": 3})
        buf.push({"n": 4})  # evicts event 1

        events, gap = buf.get_since(0)
        # since=0 never triggers gap (requesting all)
        assert gap is False
        assert len(events) == 3

        events, gap = buf.get_since(1)
        # since=1, but oldest is 2, so no gap (since+1 == oldest)
        assert gap is False

        buf.push({"n": 5})  # evicts event 2, oldest is now 3
        events, gap = buf.get_since(1)
        # since=1, oldest is 3 → gap (events 2 was evicted)
        assert gap is True
        assert len(events) == 3  # events 3, 4, 5

    def test_get_since_empty_buffer(self) -> None:
        """Empty buffer returns empty list, no gap."""
        buf = EventBuffer(capacity=100)
        events, gap = buf.get_since(0)
        assert events == []
        assert gap is False

    @pytest.mark.asyncio
    async def test_subscribe_replays_and_streams(self) -> None:
        """subscribe() replays existing events, then streams new ones."""
        buf = EventBuffer(capacity=100)
        buf.push({"n": 1})
        buf.push({"n": 2})

        received: list[int] = []

        async def consumer() -> None:
            async for event in buf.subscribe(since_id=0):
                received.append(event.id)
                if len(received) >= 4:
                    break

        task = asyncio.create_task(consumer())

        # Give consumer time to replay buffered events
        await asyncio.sleep(0.05)

        # Push new events
        buf.push({"n": 3})
        buf.push({"n": 4})

        await asyncio.wait_for(task, timeout=2.0)
        assert received == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_subscribe_with_since(self) -> None:
        """subscribe(since_id=1) skips event 1."""
        buf = EventBuffer(capacity=100)
        buf.push({"n": 1})
        buf.push({"n": 2})

        received: list[int] = []

        async def consumer() -> None:
            async for event in buf.subscribe(since_id=1):
                received.append(event.id)
                if len(received) >= 1:
                    break

        task = asyncio.create_task(consumer())
        await asyncio.wait_for(task, timeout=2.0)
        assert received == [2]

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self) -> None:
        """Multiple subscribers each get their own events independently."""
        buf = EventBuffer(capacity=100)

        received_a: list[int] = []
        received_b: list[int] = []

        async def consumer_a() -> None:
            async for event in buf.subscribe(since_id=0):
                received_a.append(event.id)
                if len(received_a) >= 2:
                    break

        async def consumer_b() -> None:
            async for event in buf.subscribe(since_id=0):
                received_b.append(event.id)
                if len(received_b) >= 2:
                    break

        task_a = asyncio.create_task(consumer_a())
        task_b = asyncio.create_task(consumer_b())

        await asyncio.sleep(0.05)
        buf.push({"n": 1})
        buf.push({"n": 2})

        await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=2.0)
        assert received_a == [1, 2]
        assert received_b == [1, 2]

    def test_thread_safety(self) -> None:
        """push() is thread-safe (uses threading lock)."""
        import threading

        buf = EventBuffer(capacity=10_000)

        def pusher(start: int, count: int) -> None:
            for i in range(count):
                buf.push({"n": start + i})

        threads = [threading.Thread(target=pusher, args=(i * 100, 100)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert buf.size == 1000
        assert buf.max_id == 1000
