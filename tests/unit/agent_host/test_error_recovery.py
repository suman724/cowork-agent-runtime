"""Tests for ErrorRecovery — failure tracking, loop detection, reflection prompts."""

from __future__ import annotations

from agent_host.loop.error_recovery import ErrorRecovery


class TestConsecutiveFailures:
    def test_no_failures_initially(self) -> None:
        er = ErrorRecovery()
        assert not er.should_inject_reflection()

    def test_reflection_after_threshold(self) -> None:
        er = ErrorRecovery(consecutive_failure_threshold=3)
        er.record_tool_failure("ReadFile", {"path": "/a"}, "not found")
        er.record_tool_failure("ReadFile", {"path": "/b"}, "not found")
        assert not er.should_inject_reflection()
        er.record_tool_failure("ReadFile", {"path": "/c"}, "not found")
        assert er.should_inject_reflection()

    def test_success_resets_counter(self) -> None:
        er = ErrorRecovery(consecutive_failure_threshold=2)
        er.record_tool_failure("ReadFile", {"path": "/a"}, "error")
        er.record_tool_success("ReadFile")
        er.record_tool_failure("ReadFile", {"path": "/b"}, "error")
        assert not er.should_inject_reflection()

    def test_reflection_prompt_content(self) -> None:
        er = ErrorRecovery(consecutive_failure_threshold=2)
        er.record_tool_failure("ReadFile", {"path": "/x"}, "File not found")
        er.record_tool_failure("WriteFile", {"path": "/y"}, "Permission denied")
        prompt = er.build_reflection_prompt()
        assert "Self-Reflection Required" in prompt
        assert "ReadFile" in prompt
        assert "WriteFile" in prompt
        assert "File not found" in prompt
        assert "Permission denied" in prompt


class TestLoopDetection:
    def test_no_loop_initially(self) -> None:
        er = ErrorRecovery()
        assert not er.detect_loop()

    def test_detects_loop(self) -> None:
        er = ErrorRecovery(loop_detection_threshold=3)
        args = {"path": "/same/file"}
        er.record_tool_failure("ReadFile", args, "error")
        er.record_tool_failure("ReadFile", args, "error")
        assert not er.detect_loop()
        er.record_tool_failure("ReadFile", args, "error")
        assert er.detect_loop()

    def test_different_args_no_loop(self) -> None:
        er = ErrorRecovery(loop_detection_threshold=3)
        er.record_tool_failure("ReadFile", {"path": "/a"}, "error")
        er.record_tool_failure("ReadFile", {"path": "/b"}, "error")
        er.record_tool_failure("ReadFile", {"path": "/c"}, "error")
        assert not er.detect_loop()

    def test_loop_break_prompt_content(self) -> None:
        er = ErrorRecovery(loop_detection_threshold=2)
        args = {"path": "/stuck"}
        er.record_tool_failure("ReadFile", args, "error")
        er.record_tool_failure("ReadFile", args, "error")
        prompt = er.build_loop_break_prompt()
        assert "Loop Detected" in prompt
        assert "different approach" in prompt

    def test_loop_priority_over_reflection(self) -> None:
        """Loop detection takes priority over reflection in the agent loop."""
        er = ErrorRecovery(consecutive_failure_threshold=2, loop_detection_threshold=2)
        args = {"path": "/same"}
        er.record_tool_failure("ReadFile", args, "error")
        er.record_tool_failure("ReadFile", args, "error")
        # Both thresholds met — loop detection takes priority in the agent loop
        assert er.should_inject_reflection()
        assert er.detect_loop()


class TestReset:
    def test_reset_clears_state(self) -> None:
        er = ErrorRecovery(consecutive_failure_threshold=1, loop_detection_threshold=1)
        er.record_tool_failure("ReadFile", {"path": "/a"}, "error")
        assert er.should_inject_reflection()
        assert er.detect_loop()

        er.reset()
        assert not er.should_inject_reflection()
        assert not er.detect_loop()
