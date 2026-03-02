"""ErrorRecovery — detects loops and injects reflection prompts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any


class ErrorRecovery:
    """Tracks tool failures and detects loops to help the agent recover.

    Monitors consecutive tool failures and repeated tool+arguments patterns.
    Generates reflection/loop-break prompts for injection into the conversation.
    """

    def __init__(
        self,
        consecutive_failure_threshold: int = 3,
        loop_detection_threshold: int = 3,
    ) -> None:
        self._consecutive_failure_threshold = consecutive_failure_threshold
        self._loop_detection_threshold = loop_detection_threshold
        self._consecutive_failures: int = 0
        self._failure_history: list[dict[str, Any]] = []
        self._call_signatures: Counter[str] = Counter()

    def record_tool_failure(self, tool_name: str, arguments: dict[str, Any], error: str) -> None:
        """Record a tool call that failed."""
        self._consecutive_failures += 1
        self._failure_history.append(
            {"tool_name": tool_name, "arguments": arguments, "error": error}
        )
        sig = self._signature(tool_name, arguments)
        self._call_signatures[sig] += 1

    def record_tool_success(self, _tool_name: str) -> None:
        """Record a successful tool call — resets consecutive failure counter."""
        self._consecutive_failures = 0

    def should_inject_reflection(self) -> bool:
        """True if consecutive failures exceed threshold."""
        return self._consecutive_failures >= self._consecutive_failure_threshold

    def detect_loop(self) -> bool:
        """True if the same tool+arguments combination has been called N+ times."""
        return any(
            count >= self._loop_detection_threshold for count in self._call_signatures.values()
        )

    def build_reflection_prompt(self) -> str:
        """Build a reflection prompt based on recent failures."""
        recent = self._failure_history[-self._consecutive_failure_threshold :]
        lines = [
            "## Self-Reflection Required",
            "",
            f"The last {len(recent)} tool calls failed. "
            "Stop and think about what went wrong before trying again.",
            "",
        ]
        for i, f in enumerate(recent, 1):
            lines.append(
                f"{i}. {f['tool_name']}({json.dumps(f['arguments'], default=str)[:200]}) "
                f"→ {f['error'][:200]}"
            )
        lines.extend(
            [
                "",
                "Consider:",
                "- Is the approach correct, or should you try a different strategy?",
                "- Are the arguments valid (correct paths, commands, etc.)?",
                "- Is there a prerequisite step you missed?",
            ]
        )
        return "\n".join(lines)

    def build_loop_break_prompt(self) -> str:
        """Build a prompt to break out of a detected loop."""
        # Find the looping signature
        looping = [
            sig
            for sig, count in self._call_signatures.items()
            if count >= self._loop_detection_threshold
        ]
        lines = [
            "## Loop Detected",
            "",
            "You appear to be repeating the same tool calls without making progress.",
            "",
        ]
        for sig in looping[:3]:
            count = self._call_signatures[sig]
            lines.append(f"- Called {count} times: {sig[:200]}")
        lines.extend(
            [
                "",
                "You MUST try a different approach. Options:",
                "- Use a different tool or different arguments",
                "- Break the task into smaller subtasks",
                "- Report what you've tried and ask the user for guidance",
            ]
        )
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._consecutive_failures = 0
        self._failure_history.clear()
        self._call_signatures.clear()

    @staticmethod
    def _signature(tool_name: str, arguments: dict[str, Any]) -> str:
        """Create a hash signature for a tool call."""
        args_str = json.dumps(arguments, sort_keys=True, default=str)
        h = hashlib.md5(args_str.encode(), usedforsecurity=False).hexdigest()[:8]
        return f"{tool_name}({h})"
