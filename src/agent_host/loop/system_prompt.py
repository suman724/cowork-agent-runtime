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
- If you encounter an error, try to recover or suggest a fix.
- ALWAYS use absolute paths for all file operations. Never use relative paths.
- When you need to install Python packages, use pip via RunCommand before importing them.
- For complex multi-step tasks, consider entering plan mode first (EnterPlanMode)
  to explore and create a structured plan before making changes.
  Exit plan mode (ExitPlanMode) when ready to execute your plan.
- When in plan mode (either via EnterPlanMode or when the task was started with
  planOnly=true), you MUST call CreatePlan to produce a structured plan. Use
  read-only tools to explore the codebase, then call CreatePlan with a clear goal
  and concrete steps. This is essential — without calling CreatePlan, the user
  will not see your plan in the UI.
- When following a plan, call UpdatePlanStep to mark each step as
  in_progress when you start it and completed when you finish it."""


class SystemPromptBuilder:
    """Builds the system prompt with workspace context and dynamic injections."""

    def __init__(self, workspace_dir: str | None = None, os_family: str | None = None) -> None:
        self._workspace_dir = workspace_dir
        self._os_family = os_family or platform.system()
        self._workspace_context = self._detect_workspace(workspace_dir)

    def build_static_prompt(
        self,
        policy_enforcer: PolicyEnforcer | None = None,
        project_instructions: str = "",
        has_persistent_memory: bool = False,
    ) -> str:
        """Build the static system prompt (identity + guidelines + workspace context).

        Args:
            policy_enforcer: Used for capability-dependent guidance.
            project_instructions: Concatenated COWORK.md content (injected after workspace).
            has_persistent_memory: Whether auto-memory is available (adds usage guidance).
        """
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

        # Project instructions (COWORK.md files)
        if project_instructions:
            parts.append(f"\n# Project Instructions\n\n{project_instructions}")

        # Memory guidance (when auto-memory is available)
        if has_persistent_memory:
            parts.append(
                "\n# Persistent Memory\n\n"
                "You have persistent memory that survives across sessions via markdown files.\n\n"
                "## Tools\n"
                "- **SaveMemory** — write/update a memory file (default: MEMORY.md)\n"
                "- **RecallMemory** — read a specific topic file\n"
                "- **ListMemories** — list all memory files\n"
                "- **DeleteMemory** — remove an outdated topic file\n\n"
                "## MEMORY.md (Index File)\n"
                "Loaded automatically every turn. Keep it concise (under 200 lines).\n"
                "Use it as an index — link to topic files for detailed notes.\n\n"
                "## Topic Files\n"
                "Create separate files (e.g., `patterns.md`, `debugging.md`) for detailed notes.\n"
                "These are read on-demand via RecallMemory.\n\n"
                "## What to Save\n"
                "- User preferences for workflow, tools, and communication style\n"
                "- Key architectural decisions, important file paths, project structure\n"
                "- Stable patterns and conventions confirmed across interactions\n"
                "- Solutions to recurring problems and debugging insights\n\n"
                "## What NOT to Save\n"
                "- Session-specific context (current task details, in-progress work)\n"
                "- Information that duplicates project instructions (COWORK.md)\n"
                "- Speculative conclusions — verify before saving\n"
                "- Sensitive data (tokens, passwords, API keys)\n\n"
                "## Rules\n"
                "- Check existing memories before writing — update, don't duplicate\n"
                "- When the user says 'remember X', save it immediately\n"
                "- When the user says 'forget X', find and remove the relevant entry\n"
                "- Organize semantically by topic, not chronologically\n"
            )

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
