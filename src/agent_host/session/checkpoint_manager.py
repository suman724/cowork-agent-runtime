"""CheckpointManager — our checkpoint format replacing ADK's CheckpointSessionService."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class SessionCheckpoint:
    """Serializable checkpoint for crash recovery."""

    session_id: str
    workspace_id: str
    tenant_id: str
    user_id: str
    token_input_used: int = 0
    token_output_used: int = 0
    session_messages: list[dict[str, Any]] = field(default_factory=list)
    thread: list[dict[str, Any]] | None = None
    working_memory: dict[str, Any] | None = None  # Wave 3
    checkpointed_at: str = ""

    # In-progress task state (all with defaults for backward compat)
    active_task_id: str | None = None  # Non-None = task in progress
    active_task_prompt: str | None = None  # The user prompt
    active_task_step: int = 0  # Last completed step
    active_task_max_steps: int = 0  # Max steps for this task
    last_workspace_sync_step: int = 0  # Last step that synced to workspace


class CheckpointManager:
    """Manages session checkpoints using atomic JSON file writes.

    Replaces ADK's CheckpointSessionService with our own format.
    Same atomic write pattern (tempfile + os.replace).
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, session_id: str) -> Path:
        """Return the checkpoint file path for a session."""
        safe_name = f"cowork_{session_id}.json"
        return self._checkpoint_dir / safe_name

    def save(self, checkpoint: SessionCheckpoint) -> None:
        """Persist a checkpoint atomically (tempfile + os.replace)."""
        path = self._checkpoint_path(checkpoint.session_id)
        checkpoint.checkpointed_at = datetime.now(tz=UTC).isoformat()

        data = asdict(checkpoint)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(self._checkpoint_dir), suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f)
                Path(tmp_path).replace(path)
            except BaseException:
                tmp = Path(tmp_path)
                if tmp.exists():
                    tmp.unlink()
                raise
        except Exception:
            logger.warning(
                "checkpoint_write_failed",
                session_id=checkpoint.session_id,
                exc_info=True,
            )

    def load(self, session_id: str) -> SessionCheckpoint | None:
        """Load a checkpoint from disk. Returns None if not found or corrupt."""
        path = self._checkpoint_path(session_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return SessionCheckpoint(
                session_id=data["session_id"],
                workspace_id=data.get("workspace_id", ""),
                tenant_id=data.get("tenant_id", ""),
                user_id=data.get("user_id", ""),
                token_input_used=data.get("token_input_used", 0),
                token_output_used=data.get("token_output_used", 0),
                session_messages=data.get("session_messages", []),
                thread=data.get("thread"),
                working_memory=data.get("working_memory"),
                checkpointed_at=data.get("checkpointed_at", ""),
                active_task_id=data.get("active_task_id"),
                active_task_prompt=data.get("active_task_prompt"),
                active_task_step=data.get("active_task_step", 0),
                active_task_max_steps=data.get("active_task_max_steps", 0),
                last_workspace_sync_step=data.get("last_workspace_sync_step", 0),
            )
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            logger.warning(
                "checkpoint_corrupt_deleting",
                session_id=session_id,
                path=str(path),
                exc_info=True,
            )
            path.unlink(missing_ok=True)
            return None

    def delete(self, session_id: str) -> None:
        """Delete a session's checkpoint file."""
        path = self._checkpoint_path(session_id)
        path.unlink(missing_ok=True)
