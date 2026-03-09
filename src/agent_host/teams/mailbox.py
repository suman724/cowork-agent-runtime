"""MailboxRouter — per-agent message queues for peer-to-peer and broadcast messaging."""

from __future__ import annotations

import asyncio

from agent_host.teams.models import TeamMessage


class MailboxRouter:
    """Routes messages between agents via per-agent asyncio queues."""

    def __init__(self) -> None:
        self._inboxes: dict[str, asyncio.Queue[TeamMessage]] = {}

    def register(self, agent_name: str) -> None:
        """Create an inbox for an agent. Idempotent."""
        if agent_name not in self._inboxes:
            self._inboxes[agent_name] = asyncio.Queue()

    def unregister(self, agent_name: str) -> None:
        """Remove an agent's inbox. Idempotent."""
        self._inboxes.pop(agent_name, None)

    async def send(self, from_agent: str, to_agent: str, content: str) -> None:
        """Send a point-to-point message."""
        queue = self._inboxes.get(to_agent)
        if queue is None:
            msg = f"Agent not registered: {to_agent}"
            raise KeyError(msg)
        message = TeamMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            message_type="message",
        )
        queue.put_nowait(message)

    async def broadcast(self, from_agent: str, content: str) -> None:
        """Send a message to all registered agents except the sender."""
        for name, queue in self._inboxes.items():
            if name == from_agent:
                continue
            message = TeamMessage(
                from_agent=from_agent,
                to_agent=name,
                content=content,
                message_type="broadcast",
            )
            queue.put_nowait(message)

    async def poll(
        self,
        agent_name: str,
        timeout: float = 0.0,
    ) -> list[TeamMessage]:
        """Retrieve all pending messages for an agent.

        If timeout=0: non-blocking, returns whatever is queued.
        If timeout>0: waits up to timeout seconds for at least one message,
        then drains all available messages.
        """
        queue = self._inboxes.get(agent_name)
        if queue is None:
            msg = f"Agent not registered: {agent_name}"
            raise KeyError(msg)

        messages: list[TeamMessage] = []

        if queue.empty() and timeout > 0:
            try:
                first = await asyncio.wait_for(queue.get(), timeout=timeout)
                messages.append(first)
            except TimeoutError:
                return []

        # Drain remaining
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        return messages

    def has_messages(self, agent_name: str) -> bool:
        """Check if an agent has pending messages (non-blocking)."""
        queue = self._inboxes.get(agent_name)
        if queue is None:
            return False
        return not queue.empty()
