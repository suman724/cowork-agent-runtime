"""CheckpointManager — our checkpoint format replacing ADK's CheckpointSessionService."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent_host.coordination.protocols import CheckpointStrategy

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

    # Workspace directory path for memory restoration on crash recovery
    workspace_dir: str | None = None

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

    def __init__(
        self,
        checkpoint_dir: str,
        extra_strategies: list[CheckpointStrategy] | None = None,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._extra_strategies = extra_strategies or []

    def _checkpoint_path(self, session_id: str) -> Path:
        """Return the checkpoint file path for a session."""
        safe_name = f"cowork_{session_id}.json"
        return self._checkpoint_dir / safe_name

    def save(self, checkpoint: SessionCheckpoint) -> None:
        """Persist a checkpoint atomically (tempfile + os.replace).

        Also captures state from any extra checkpoint strategies
        and stores it under the ``strategy_state`` key.
        """
        path = self._checkpoint_path(checkpoint.session_id)
        checkpoint.checkpointed_at = datetime.now(tz=UTC).isoformat()

        data = asdict(checkpoint)

        # Capture extra strategy state (e.g. team coordination metadata)
        if self._extra_strategies:
            strategy_state: dict[str, Any] = {}
            for i, strategy in enumerate(self._extra_strategies):
                key = f"strategy_{i}"
                try:
                    strategy_state[key] = strategy.capture()
                except Exception:
                    logger.warning("strategy_capture_failed", strategy_index=i, exc_info=True)
            if strategy_state:
                data["strategy_state"] = strategy_state
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
        """Load a checkpoint from disk. Returns None if not found or corrupt.

        Strategy state from extra checkpoint strategies is stored in the
        returned checkpoint's raw data but not in the dataclass fields.
        Call :meth:`restore_strategies` after loading to restore strategy state.
        """
        path = self._checkpoint_path(session_id)
        if not path.exists():
            return None

        try:
            raw = json.loads(path.read_text())
            checkpoint = SessionCheckpoint(
                session_id=raw["session_id"],
                workspace_id=raw.get("workspace_id", ""),
                tenant_id=raw.get("tenant_id", ""),
                user_id=raw.get("user_id", ""),
                token_input_used=raw.get("token_input_used", 0),
                token_output_used=raw.get("token_output_used", 0),
                session_messages=raw.get("session_messages", []),
                thread=raw.get("thread"),
                working_memory=raw.get("working_memory"),
                checkpointed_at=raw.get("checkpointed_at", ""),
                workspace_dir=raw.get("workspace_dir"),
                active_task_id=raw.get("active_task_id"),
                active_task_prompt=raw.get("active_task_prompt"),
                active_task_step=raw.get("active_task_step", 0),
                active_task_max_steps=raw.get("active_task_max_steps", 0),
                last_workspace_sync_step=raw.get("last_workspace_sync_step", 0),
            )
            # Stash raw strategy state for restore_strategies()
            self._loaded_strategy_state = raw.get("strategy_state", {})
            return checkpoint
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            logger.warning(
                "checkpoint_corrupt_deleting",
                session_id=session_id,
                path=str(path),
                exc_info=True,
            )
            path.unlink(missing_ok=True)
            return None

    async def restore_strategies(self) -> None:
        """Restore extra strategy state from the last loaded checkpoint.

        Must be called after :meth:`load`. No-op if no strategy state was found.
        """
        strategy_state = getattr(self, "_loaded_strategy_state", {})
        for i, strategy in enumerate(self._extra_strategies):
            key = f"strategy_{i}"
            state = strategy_state.get(key, {})
            if state:
                try:
                    await strategy.restore(state)
                except Exception:
                    logger.warning("strategy_restore_failed", strategy_index=i, exc_info=True)

    def delete(self, session_id: str) -> None:
        """Delete a session's checkpoint file."""
        path = self._checkpoint_path(session_id)
        path.unlink(missing_ok=True)
