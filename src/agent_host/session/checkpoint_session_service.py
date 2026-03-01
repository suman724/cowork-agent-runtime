"""ADK SessionService backed by atomic JSON file checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import structlog
from google.adk.events import Event  # noqa: TC002
from google.adk.sessions import Session
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)

logger = structlog.get_logger()


class CheckpointSessionService(BaseSessionService):
    """ADK SessionService backed by JSON file persistence.

    - create_session() → creates in-memory + writes checkpoint
    - get_session() → reads from memory (loaded from checkpoint on startup)
    - append_event() → appends to session, writes checkpoint atomically
    - Atomic write: tempfile + os.replace()
    - Corruption handling: delete corrupt file, return None
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}

    def _checkpoint_path(self, app_name: str, session_id: str) -> Path:
        """Return the checkpoint file path for a session."""
        safe_name = f"{app_name}_{session_id}.json"
        return self._checkpoint_dir / safe_name

    def _write_checkpoint(self, app_name: str, session: Session) -> None:
        """Write session state to disk atomically."""
        path = self._checkpoint_path(app_name, session.id)
        data = {
            "id": session.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
            "state": dict(session.state) if session.state else {},
            "last_update_time": session.last_update_time,
        }
        try:
            # Atomic write: write to temp file, then rename
            fd, tmp_path = tempfile.mkstemp(dir=str(self._checkpoint_dir), suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f)
                Path(tmp_path).replace(path)
            except BaseException:
                # Clean up temp file on failure
                with_path = Path(tmp_path)
                if with_path.exists():
                    with_path.unlink()
                raise
        except Exception:
            logger.warning(
                "checkpoint_write_failed",
                session_id=session.id,
                exc_info=True,
            )

    def _load_checkpoint(self, app_name: str, session_id: str) -> Session | None:
        """Load a session from its checkpoint file."""
        path = self._checkpoint_path(app_name, session_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            session = Session(
                id=data["id"],
                app_name=data.get("app_name", app_name),
                user_id=data.get("user_id", ""),
                state=data.get("state", {}),
                events=[],
                last_update_time=data.get("last_update_time", 0.0),
            )
            return session
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            logger.warning(
                "checkpoint_corrupt_deleting",
                session_id=session_id,
                path=str(path),
                exc_info=True,
            )
            path.unlink(missing_ok=True)
            return None

    def _delete_checkpoint(self, app_name: str, session_id: str) -> None:
        """Delete a session's checkpoint file."""
        path = self._checkpoint_path(app_name, session_id)
        path.unlink(missing_ok=True)

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Create a new session and write its initial checkpoint."""
        import time

        sid = session_id or f"session-{os.urandom(8).hex()}"
        session = Session(
            id=sid,
            app_name=app_name,
            user_id=user_id,
            state=state or {},
            events=[],
            last_update_time=time.time(),
        )
        self._sessions[sid] = session
        self._write_checkpoint(app_name, session)
        return session

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: GetSessionConfig | None = None,
    ) -> Session | None:
        """Get a session by ID. Falls back to checkpoint if not in memory."""
        session = self._sessions.get(session_id)
        if session is not None:
            return session

        # Try loading from checkpoint
        session = self._load_checkpoint(app_name, session_id)
        if session is not None:
            self._sessions[session_id] = session
        return session

    async def list_sessions(
        self,
        *,
        app_name: str,
        user_id: str | None = None,
    ) -> ListSessionsResponse:
        """List all in-memory sessions."""
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        return ListSessionsResponse(sessions=sessions)

    async def delete_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        """Delete a session and its checkpoint."""
        self._sessions.pop(session_id, None)
        self._delete_checkpoint(app_name, session_id)

    async def append_event(self, session: Session, event: Event) -> Event:
        """Append an event to the session and write a checkpoint."""
        # Apply state delta if present
        if event.actions and event.actions.state_delta:
            for key, value in event.actions.state_delta.items():
                session.state[key] = value

        session.events.append(event)

        import time

        session.last_update_time = time.time()

        self._write_checkpoint(session.app_name, session)
        return event
