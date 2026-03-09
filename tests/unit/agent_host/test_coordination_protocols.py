"""Tests for coordination strategy protocols and Solo implementations."""

from __future__ import annotations

import pytest

from agent_host.coordination.protocols import (
    CheckpointStrategy,
    ContextInjectionStrategy,
    CoordinationStrategy,
    ToolProviderStrategy,
)
from agent_host.coordination.solo import (
    SoloCheckpointProvider,
    SoloContextInjector,
    SoloCoordinator,
    SoloToolProvider,
)

# ── Protocol conformance ──────────────────────────────────────────


class TestProtocolConformance:
    """Verify Solo implementations satisfy their protocols (runtime_checkable)."""

    def test_solo_coordinator_satisfies_protocol(self) -> None:
        assert isinstance(SoloCoordinator(), CoordinationStrategy)

    def test_solo_tool_provider_satisfies_protocol(self) -> None:
        assert isinstance(SoloToolProvider(), ToolProviderStrategy)

    def test_solo_context_injector_satisfies_protocol(self) -> None:
        assert isinstance(SoloContextInjector(), ContextInjectionStrategy)

    def test_solo_checkpoint_provider_satisfies_protocol(self) -> None:
        assert isinstance(SoloCheckpointProvider(), CheckpointStrategy)


# ── SoloCoordinator ─────────────────────────────────────────────


class TestSoloCoordinator:
    @pytest.mark.asyncio
    async def test_on_session_start_is_noop(self) -> None:
        c = SoloCoordinator()
        await c.on_session_start("sess_1", {})  # should not raise

    @pytest.mark.asyncio
    async def test_on_session_shutdown_is_noop(self) -> None:
        c = SoloCoordinator()
        await c.on_session_shutdown()  # should not raise

    @pytest.mark.asyncio
    async def test_spawn_agent_raises(self) -> None:
        c = SoloCoordinator()
        with pytest.raises(RuntimeError, match="coordination strategy"):
            await c.spawn_agent("test", "role", "prompt", 1000)

    @pytest.mark.asyncio
    async def test_shutdown_agent_is_noop(self) -> None:
        c = SoloCoordinator()
        await c.shutdown_agent("test")

    @pytest.mark.asyncio
    async def test_shutdown_all_is_noop(self) -> None:
        c = SoloCoordinator()
        await c.shutdown_all()

    def test_get_active_agents_returns_empty(self) -> None:
        c = SoloCoordinator()
        assert c.get_active_agents() == []


# ── SoloToolProvider ─────────────────────────────────────────────


class TestSoloToolProvider:
    def test_get_tool_definitions_returns_empty(self) -> None:
        p = SoloToolProvider()
        assert p.get_tool_definitions("solo") == []
        assert p.get_tool_definitions("lead") == []
        assert p.get_tool_definitions("teammate") == []

    @pytest.mark.asyncio
    async def test_handle_tool_call_raises(self) -> None:
        p = SoloToolProvider()
        with pytest.raises(RuntimeError, match="does not handle"):
            await p.handle_tool_call("SomeTool", {}, "agent")

    def test_owns_tool_returns_false(self) -> None:
        p = SoloToolProvider()
        assert p.owns_tool("CreateTeam") is False
        assert p.owns_tool("TaskTracker") is False


# ── SoloContextInjector ─────────────────────────────────────────


class TestSoloContextInjector:
    @pytest.mark.asyncio
    async def test_get_injections_returns_empty(self) -> None:
        inj = SoloContextInjector()
        assert await inj.get_injections("lead") == []
        assert await inj.get_injections("teammate") == []

    def test_estimate_overhead_tokens_returns_zero(self) -> None:
        inj = SoloContextInjector()
        assert inj.estimate_overhead_tokens() == 0


# ── SoloCheckpointProvider ───────────────────────────────────────


class TestSoloCheckpointProvider:
    def test_capture_returns_empty_dict(self) -> None:
        cp = SoloCheckpointProvider()
        assert cp.capture() == {}

    @pytest.mark.asyncio
    async def test_restore_is_noop(self) -> None:
        cp = SoloCheckpointProvider()
        await cp.restore({"some": "data"})  # should not raise
