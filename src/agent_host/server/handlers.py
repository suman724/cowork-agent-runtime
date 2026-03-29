"""JSON-RPC method handlers — thin delegation to SessionManager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_host.session.session_manager import SessionManager
    from agent_host.transport.method_dispatcher import MethodDispatcher


class Handlers:
    """Registers JSON-RPC method handlers that delegate to SessionManager.

    Methods:
        CreateSession — Initialize session with Session Service
        ResumeSession — Resume an existing session (reuses session ID)
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
        dispatcher.register("ResumeSession", self.handle_resume_session)
        dispatcher.register("StartTask", self.handle_start_task)
        dispatcher.register("CancelTask", self.handle_cancel_task)
        dispatcher.register("GetSessionState", self.handle_get_session_state)
        dispatcher.register("ApproveAction", self.handle_approve_action)
        dispatcher.register("GetPatchPreview", self.handle_get_patch_preview)
        dispatcher.register("Shutdown", self.handle_shutdown)
        dispatcher.register("GetEvents", self.handle_get_events)
        dispatcher.register("browser.pause", self.handle_browser_pause)
        dispatcher.register("browser.resume", self.handle_browser_resume)

    async def handle_create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """CreateSession — initialize session with Session Service."""
        return await self._session_manager.create_session(params)

    async def handle_resume_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """ResumeSession — resume an existing session, reusing the same session ID."""
        return await self._session_manager.resume_session(params)

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

    async def handle_get_events(self, params: dict[str, Any]) -> dict[str, Any]:
        """GetEvents — return buffered events since a given ID.

        Params:
            sinceId (int): Return events with ID > sinceId (default: 0 = all)

        Returns:
            events: list of event dicts with eventId
            gapDetected: True if some events were evicted (data loss)
            latestId: highest event ID in the buffer
        """
        return self._session_manager.get_events(params)

    async def handle_shutdown(self, params: dict[str, Any]) -> dict[str, Any]:
        """Shutdown — clean session teardown."""
        return await self._session_manager.shutdown()

    async def handle_browser_pause(self, params: dict[str, Any]) -> dict[str, Any]:
        """browser.pause — pause agent for user takeover of headed browser."""
        browser_mgr = getattr(self._session_manager._tool_router, "_browser_manager", None)
        if browser_mgr is None:
            return {"status": "error", "message": "Browser not available"}
        browser_mgr.pause()
        return {"status": "paused"}

    async def handle_browser_resume(self, params: dict[str, Any]) -> dict[str, Any]:
        """browser.resume — resume agent after user takeover."""
        browser_mgr = getattr(self._session_manager._tool_router, "_browser_manager", None)
        if browser_mgr is None:
            return {"status": "error", "message": "Browser not available"}
        browser_mgr.resume()
        return {"status": "resumed"}
