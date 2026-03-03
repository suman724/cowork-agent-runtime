"""Dynamic system prompt builder for the agent loop."""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_host.policy.policy_enforcer import PolicyEnforcer

# Base system prompt — same as the prior SYSTEM_PROMPT from agent_factory.py
_BASE_SYSTEM_PROMPT = """You are Cowork, a capable AI assistant running on the user's desktop.
You have access to tools for reading files, writing files, deleting files,
running shell commands, and making HTTP requests.

Guidelines:
- Use tools to accomplish the user's requests. Always verify your work.
- When writing files, show the user what you plan to write before doing so.
- When running commands, explain what the command does.
- If a tool call is denied by policy, explain why and suggest alternatives.
- Be concise and helpful. Focus on completing the task efficiently.
- If you encounter an error, try to recover or suggest a fix."""


class SystemPromptBuilder:
    """Builds the system prompt with workspace context and dynamic injections."""

    def __init__(self, workspace_dir: str | None = None, os_family: str | None = None) -> None:
        self._workspace_dir = workspace_dir
        self._os_family = os_family or platform.system()
        self._workspace_context = self._detect_workspace(workspace_dir)

    def build_static_prompt(self, policy_enforcer: PolicyEnforcer | None = None) -> str:
        """Build the static system prompt (identity + guidelines + workspace context)."""
        parts = [_BASE_SYSTEM_PROMPT]

        # Current date
        parts.append(f"\nCurrent date: {datetime.now(tz=UTC).strftime('%Y-%m-%d')}")

        # OS context
        parts.append(f"\nOperating system: {self._os_family}")

        # Workspace context
        if self._workspace_context:
            parts.append(f"\nWorkspace:\n{self._workspace_context}")

        # Capability-dependent guidance
        if policy_enforcer is not None:
            cap_guidance = self._build_capability_guidance(policy_enforcer)
            if cap_guidance:
                parts.append(cap_guidance)

        return "\n".join(parts)

    def build_dynamic_injection(
        self,
        working_memory: object | None = None,
        error_recovery: object | None = None,
    ) -> str:
        """Build per-turn dynamic injection (working memory, error state).

        Returns empty string if no dynamic content to inject.
        Placeholders for Wave 3+ (working memory) and Wave 5 (error recovery).
        """
        parts: list[str] = []

        # Wave 3: Working memory injection
        if working_memory is not None and hasattr(working_memory, "render"):
            rendered = working_memory.render()
            if rendered:
                parts.append(rendered)

        # Wave 5: Error recovery injection
        if (
            error_recovery is not None
            and hasattr(error_recovery, "build_reflection_prompt")
            and hasattr(error_recovery, "should_inject_reflection")
            and error_recovery.should_inject_reflection()
        ):
            parts.append(error_recovery.build_reflection_prompt())

        return "\n\n".join(parts)

    @staticmethod
    def _build_capability_guidance(policy_enforcer: PolicyEnforcer) -> str:
        """Build guidance text based on granted capabilities."""
        parts: list[str] = []

        from cowork_platform_sdk import CapabilityName

        if policy_enforcer.get_capability(CapabilityName.CODE_EXECUTE) is not None:
            parts.append(
                "\nYou have access to the ExecuteCode tool for running Python scripts.\n"
                "Use it for: data analysis, calculations, testing code you've written, "
                "prototyping.\n"
                "Each execution is independent — write complete, self-contained scripts.\n"
                "You can generate plots with matplotlib (plt.show() saves images "
                "automatically).\n"
                "For file operations on the workspace, prefer the dedicated file tools "
                "(ReadFile, WriteFile)."
            )

        return "\n".join(parts)

    @staticmethod
    def _detect_workspace(workspace_dir: str | None) -> str:
        """Detect project type and key files in the workspace directory."""
        if not workspace_dir:
            return ""
        ws_path = Path(workspace_dir)
        if not ws_path.is_dir():
            return ""

        context_parts: list[str] = [f"Directory: {workspace_dir}"]

        # Check for common project indicators
        indicators = {
            "pyproject.toml": "Python project (pyproject.toml)",
            "package.json": "Node.js project (package.json)",
            "Cargo.toml": "Rust project (Cargo.toml)",
            "go.mod": "Go project (go.mod)",
            "Makefile": "Has Makefile",
            ".git": "Git repository",
        }

        for filename, description in indicators.items():
            if (ws_path / filename).exists():
                context_parts.append(f"- {description}")

        return "\n".join(context_parts) if len(context_parts) > 1 else ""
