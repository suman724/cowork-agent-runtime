"""Test that LoopRuntime satisfies the LoopContext protocol."""

from __future__ import annotations

from agent_sdk.loop.context import LoopContext

from agent_host.loop.loop_runtime import LoopRuntime


class TestLoopContextConformance:
    def test_loop_runtime_has_all_protocol_members(self) -> None:
        """LoopRuntime must have all LoopContext protocol methods and properties."""
        # LoopContext has non-callable members (properties), so issubclass() doesn't work.
        # Instead verify all protocol members exist on LoopRuntime.
        for name in LoopContext.__protocol_attrs__:
            assert hasattr(LoopRuntime, name), f"LoopRuntime missing LoopContext member: {name}"
