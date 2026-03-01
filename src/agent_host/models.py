"""Internal data models for the Agent Host."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SessionContext:
    """Immutable session identity — threaded through all operations."""

    session_id: str
    workspace_id: str
    tenant_id: str
    user_id: str


@dataclass(frozen=True)
class PolicyCheckResult:
    """Result of a policy capability check."""

    decision: Literal["ALLOWED", "DENIED", "APPROVAL_REQUIRED"]
    reason: str = ""
    approval_rule_id: str | None = None
    risk_level: str | None = None
