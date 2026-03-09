"""Data models for agent teams."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

AgentRole = Literal["lead", "teammate", "solo"]

TaskStatus = Literal["pending", "in_progress", "completed", "failed", "blocked"]

MessageType = Literal["message", "broadcast", "shutdown_request", "shutdown_response"]


@dataclass
class TeamConfig:
    """Configuration for a team."""

    name: str = "default"
    description: str = ""
    max_teammates: int = 5


@dataclass
class TeammateInfo:
    """Metadata for a team member."""

    name: str
    role: str
    budget: int
    status: Literal["created", "running", "shutting_down", "stopped"] = "created"
    session_id: str | None = None


@dataclass
class TeamTask:
    """A task in the shared task list."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    status: TaskStatus = "pending"
    assignee: str | None = None
    created_by: str = ""
    blocked_by: list[str] = field(default_factory=list)
    result: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass
class TeamMessage:
    """A message in the mailbox system."""

    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    from_agent: str = ""
    to_agent: str | None = None
    content: str = ""
    message_type: MessageType = "message"
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
