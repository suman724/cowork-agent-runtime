"""ApprovalGate — manages pending approval requests as asyncio Futures."""

from __future__ import annotations

import asyncio

import structlog

logger = structlog.get_logger()


class ApprovalGate:
    """Manages asyncio.Future instances keyed by approval_id.

    When a tool requires approval, tool_fn calls ``request_approval()`` which
    creates a Future and awaits the user decision.  The JSON-RPC handler for
    ``ApproveAction`` calls ``deliver()`` to resolve the Future.
    """

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[str]] = {}

    async def request_approval(self, approval_id: str, timeout: float = 300.0) -> str:
        """Create a Future and await user decision.

        Returns:
            ``"approved"`` or ``"denied"``.  On timeout, returns ``"denied"``.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending[approval_id] = future

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            logger.warning("approval_timeout", approval_id=approval_id, timeout=timeout)
            return "denied"
        finally:
            self._pending.pop(approval_id, None)

    def deliver(self, approval_id: str, decision: str) -> bool:
        """Resolve a pending approval request.

        Returns:
            ``True`` if a pending request was resolved, ``False`` otherwise.
        """
        future = self._pending.get(approval_id)
        if future is not None and not future.done():
            future.set_result(decision)
            return True
        return False

    @property
    def pending_count(self) -> int:
        """Number of approvals currently awaiting a decision."""
        return len(self._pending)
