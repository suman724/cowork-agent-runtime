"""Bounded ring buffer for SSE event replay.

Events are assigned monotonic integer IDs starting from 1. When the buffer
is full, the oldest events are evicted. SSE clients can request replay from
a specific ID using ``?since={id}``.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

DEFAULT_CAPACITY = 10_000


@dataclass(frozen=True)
class BufferedEvent:
    """An event stored in the ring buffer with a monotonic ID."""

    id: int
    data: dict[str, Any]


class EventBuffer:
    """Thread-safe bounded ring buffer for SSE events.

    - ``push()`` adds an event and notifies all waiting subscribers.
    - ``get_since(since_id)`` returns events after the given ID.
    - ``subscribe()`` returns an async generator that yields new events.

    Concurrency: ``push()`` may be called from sync contexts (event emitter
    fire-and-forget), so we use a threading lock for the buffer. Async
    notification uses an asyncio.Event per subscriber.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._capacity = capacity
        self._buffer: deque[BufferedEvent] = deque(maxlen=capacity)
        self._next_id = 1
        self._lock = threading.Lock()
        # Active subscriber notifications — each subscriber gets its own Event
        self._subscribers: list[asyncio.Event] = []

    @property
    def min_id(self) -> int:
        """Smallest event ID still in the buffer, or 0 if empty."""
        with self._lock:
            if not self._buffer:
                return 0
            return self._buffer[0].id

    @property
    def max_id(self) -> int:
        """Largest event ID in the buffer, or 0 if empty."""
        with self._lock:
            if not self._buffer:
                return 0
            return self._buffer[-1].id

    @property
    def size(self) -> int:
        """Number of events currently in the buffer."""
        with self._lock:
            return len(self._buffer)

    def push(self, event_data: dict[str, Any]) -> int:
        """Add an event to the buffer. Returns the assigned event ID.

        Thread-safe. Notifies all active subscribers.
        """
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            self._buffer.append(BufferedEvent(id=event_id, data=event_data))

        # Notify subscribers (best-effort, non-blocking)
        for subscriber_event in self._subscribers:
            subscriber_event.set()

        return event_id

    def get_since(self, since_id: int) -> tuple[list[BufferedEvent], bool]:
        """Return events with ID > since_id.

        Returns:
            Tuple of (events, gap_detected).
            - events: list of BufferedEvent with id > since_id
            - gap_detected: True if some events between since_id and the
              oldest buffered event were evicted (data loss).
              When since_id is 0, gap_detected is always False (requesting all).
        """
        with self._lock:
            if not self._buffer:
                return [], False

            oldest_id = self._buffer[0].id
            gap = since_id > 0 and since_id < oldest_id - 1

            events = [e for e in self._buffer if e.id > since_id]
            return events, gap

    async def subscribe(self, since_id: int = 0) -> Any:
        """Async generator yielding BufferedEvents as they arrive.

        First replays events after since_id, then waits for new events.
        Yields (event, gap_detected) on first batch only for gap detection.
        """
        notify = asyncio.Event()
        self._subscribers.append(notify)
        try:
            last_seen = since_id
            # Initial replay
            events, gap = self.get_since(last_seen)
            if gap:
                yield BufferedEvent(id=-1, data={"_gap": True, "since": since_id})
            for event in events:
                yield event
                last_seen = event.id

            # Live stream
            while True:
                notify.clear()
                events, _ = self.get_since(last_seen)
                for event in events:
                    yield event
                    last_seen = event.id
                await notify.wait()
        finally:
            self._subscribers.remove(notify)
