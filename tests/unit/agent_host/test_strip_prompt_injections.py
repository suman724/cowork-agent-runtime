"""Tests for SessionManager._strip_prompt_injections."""

from agent_host.session.session_manager import SessionManager


class TestStripPromptInjections:
    def test_plain_prompt_unchanged(self) -> None:
        assert SessionManager._strip_prompt_injections("Hello world") == "Hello world"

    def test_strips_workspace_prefix(self) -> None:
        raw = "[Workspace: /Users/me/project]\n\nHello world"
        assert SessionManager._strip_prompt_injections(raw) == "Hello world"

    def test_strips_plan_only_suffix(self) -> None:
        raw = (
            "Hello world\n\n[IMPORTANT: You are in plan-only mode. You MUST call the "
            "CreatePlan tool with a goal and steps before finishing. Use read-only tools "
            "to explore first, then call CreatePlan. Do NOT skip calling CreatePlan.]"
        )
        assert SessionManager._strip_prompt_injections(raw) == "Hello world"

    def test_strips_both(self) -> None:
        raw = (
            "[Workspace: /tmp/ws]\n\nDo something\n\n[IMPORTANT: You are in plan-only mode. "
            "You MUST call the CreatePlan tool with a goal and steps before finishing. "
            "Use read-only tools to explore first, then call CreatePlan. "
            "Do NOT skip calling CreatePlan.]"
        )
        assert SessionManager._strip_prompt_injections(raw) == "Do something"

    def test_empty_string(self) -> None:
        assert SessionManager._strip_prompt_injections("") == ""

    def test_multiline_prompt_preserved(self) -> None:
        raw = "[Workspace: /a/b]\n\nLine one\nLine two\nLine three"
        assert SessionManager._strip_prompt_injections(raw) == "Line one\nLine two\nLine three"
