"""Tests for ApprovalGate — async approval suspend/resume."""

from __future__ import annotations

import asyncio

import pytest

from agent_host.approval.approval_gate import ApprovalGate


class TestApprovalGate:
    @pytest.mark.asyncio
    async def test_request_and_deliver_approved(self) -> None:
        """Deliver 'approved' before timeout returns 'approved'."""
        gate = ApprovalGate()

        async def _deliver_soon() -> None:
            await asyncio.sleep(0.01)
            gate.deliver("appr-1", "approved")

        _task = asyncio.create_task(_deliver_soon())  # noqa: RUF006
        result = await gate.request_approval("appr-1", timeout=5.0)
        assert result == "approved"
        assert gate.pending_count == 0

    @pytest.mark.asyncio
    async def test_request_and_deliver_denied(self) -> None:
        """Deliver 'denied' returns 'denied'."""
        gate = ApprovalGate()

        async def _deliver_soon() -> None:
            await asyncio.sleep(0.01)
            gate.deliver("appr-2", "denied")

        _task = asyncio.create_task(_deliver_soon())  # noqa: RUF006
        result = await gate.request_approval("appr-2", timeout=5.0)
        assert result == "denied"

    @pytest.mark.asyncio
    async def test_timeout_returns_denied(self) -> None:
        """No delivery within timeout returns 'denied'."""
        gate = ApprovalGate()
        result = await gate.request_approval("appr-3", timeout=0.05)
        assert result == "denied"
        assert gate.pending_count == 0

    @pytest.mark.asyncio
    async def test_deliver_unknown_id_returns_false(self) -> None:
        """Delivering to an unknown approval ID returns False."""
        gate = ApprovalGate()
        assert gate.deliver("nonexistent", "approved") is False

    @pytest.mark.asyncio
    async def test_deliver_after_timeout_returns_false(self) -> None:
        """Delivering after the request timed out returns False."""
        gate = ApprovalGate()
        await gate.request_approval("appr-4", timeout=0.01)

        # Future is cleaned up after timeout
        assert gate.deliver("appr-4", "approved") is False

    @pytest.mark.asyncio
    async def test_concurrent_approvals(self) -> None:
        """Two parallel approval requests are resolved independently."""
        gate = ApprovalGate()

        async def _deliver_both() -> None:
            await asyncio.sleep(0.01)
            gate.deliver("a1", "approved")
            gate.deliver("a2", "denied")

        _task = asyncio.create_task(_deliver_both())  # noqa: RUF006

        r1, r2 = await asyncio.gather(
            gate.request_approval("a1", timeout=5.0),
            gate.request_approval("a2", timeout=5.0),
        )
        assert r1 == "approved"
        assert r2 == "denied"
        assert gate.pending_count == 0
