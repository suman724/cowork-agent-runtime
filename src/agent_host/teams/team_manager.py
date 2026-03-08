"""TeamManager — coordinates task list, mailbox, and teammate lifecycle."""

from __future__ import annotations

import uuid

from agent_host.teams.mailbox import MailboxRouter
from agent_host.teams.models import TeamConfig, TeammateInfo
from agent_host.teams.task_list import SharedTaskList


class TeamManager:
    """Owns shared primitives and manages teammate registration.

    Does NOT create SessionManager instances or spawn agent loops —
    that is handled at the integration layer (Increment 4).
    """

    def __init__(
        self,
        lead_session_id: str,
        config: TeamConfig | None = None,
    ) -> None:
        self.team_id: str = uuid.uuid4().hex[:12]
        self.lead_session_id = lead_session_id
        self.config = config or TeamConfig()
        self.task_list = SharedTaskList()
        self.mailbox = MailboxRouter()
        self.members: dict[str, TeammateInfo] = {}

        # Register lead in mailbox
        self.mailbox.register("lead")

    async def create_teammate(
        self,
        name: str,
        role: str,
        budget: int,
    ) -> TeammateInfo:
        """Register a new teammate. Does not spawn an agent loop."""
        if name in self.members:
            msg = f"Teammate already exists: {name}"
            raise ValueError(msg)
        if name == "lead":
            msg = "Cannot use reserved name: lead"
            raise ValueError(msg)
        if len(self.members) >= self.config.max_teammates:
            msg = f"Team is full: {len(self.members)}/{self.config.max_teammates} teammates"
            raise ValueError(msg)

        info = TeammateInfo(name=name, role=role, budget=budget)
        self.members[name] = info
        self.mailbox.register(name)
        return info

    async def shutdown_teammate(self, name: str) -> None:
        """Remove a teammate from the team."""
        if name not in self.members:
            msg = f"Teammate not found: {name}"
            raise KeyError(msg)
        self.members[name].status = "stopped"
        self.mailbox.unregister(name)
        del self.members[name]

    async def shutdown_all(self) -> None:
        """Shut down all teammates."""
        names = list(self.members.keys())
        for name in names:
            await self.shutdown_teammate(name)

    def get_active_agents(self) -> list[TeammateInfo]:
        """Return list of active teammates."""
        return list(self.members.values())

    def get_teammate(self, name: str) -> TeammateInfo | None:
        """Get teammate info by name."""
        return self.members.get(name)

    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
    ) -> None:
        """Send a point-to-point message via the mailbox."""
        await self.mailbox.send(from_agent, to_agent, content)

    async def broadcast_message(self, from_agent: str, content: str) -> None:
        """Broadcast a message to all agents via the mailbox."""
        await self.mailbox.broadcast(from_agent, content)
