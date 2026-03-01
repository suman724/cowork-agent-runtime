"""JSON-RPC method handlers — thin delegation to SessionManager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_host.server.method_dispatcher import MethodDispatcher
    from agent_host.session.session_manager import SessionManager


class Handlers:
    """Registers JSON-RPC method handlers that delegate to SessionManager.

    Methods:
        CreateSession — Initialize session with Session Service
        StartTask — Begin agent work cycle from user prompt
        CancelTask — Cooperatively cancel running task
        GetSessionState — Return current session/task status
        ApproveAction — Deliver user approval/denial decision
        GetPatchPreview — Return unified diffs for file changes
        Shutdown — Clean session teardown
    """

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    def register_all(self, dispatcher: MethodDispatcher) -> None:
        """Register all handlers with the method dispatcher."""
        dispatcher.register("CreateSession", self.handle_create_session)
        dispatcher.register("StartTask", self.handle_start_task)
        dispatcher.register("CancelTask", self.handle_cancel_task)
        dispatcher.register("GetSessionState", self.handle_get_session_state)
        dispatcher.register("ApproveAction", self.handle_approve_action)
        dispatcher.register("GetPatchPreview", self.handle_get_patch_preview)
        dispatcher.register("Shutdown", self.handle_shutdown)

    async def handle_create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """CreateSession — initialize session with Session Service."""
        return await self._session_manager.create_session(params)

    async def handle_start_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """StartTask — begin agent work cycle from user prompt."""
        return await self._session_manager.start_task(params)

    async def handle_cancel_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """CancelTask — cooperatively cancel running task."""
        return await self._session_manager.cancel_task()

    async def handle_get_session_state(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """GetSessionState — return current session/task status."""
        return await self._session_manager.get_session_state()

    async def handle_approve_action(self, params: dict[str, Any]) -> dict[str, Any]:
        """ApproveAction — deliver user approval/denial decision."""
        return await self._session_manager.deliver_approval(params)

    async def handle_get_patch_preview(self, params: dict[str, Any]) -> dict[str, Any]:
        """GetPatchPreview — return unified diffs for file changes in a task."""
        return await self._session_manager.get_patch_preview(params)

    async def handle_shutdown(self, params: dict[str, Any]) -> dict[str, Any]:
        """Shutdown — clean session teardown."""
        return await self._session_manager.shutdown()
