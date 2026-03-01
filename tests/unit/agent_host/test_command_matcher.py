"""Tests for shell command matching."""

from __future__ import annotations

from agent_host.policy.command_matcher import check_command, extract_base_command


class TestExtractBaseCommand:
    def test_simple_command(self) -> None:
        assert extract_base_command("ls -la") == "ls"

    def test_with_path_prefix(self) -> None:
        assert extract_base_command("/usr/bin/git status") == "git"

    def test_with_backslash_in_token(self) -> None:
        """Backslash path stripping works when token contains literal backslash."""
        # Use a raw string to avoid shlex consuming the backslashes
        assert extract_base_command(r'"C:\Python39\python.exe" script.py') == "python.exe"

    def test_malformed_quoting(self) -> None:
        """Malformed quoting falls back to whitespace split."""
        assert extract_base_command("echo 'unterminated") == "echo"

    def test_empty_command(self) -> None:
        assert extract_base_command("") == ""

    def test_command_with_args(self) -> None:
        assert extract_base_command("python -m pytest tests/") == "python"

    def test_command_with_chaining(self) -> None:
        """Extracts only the first command from chained commands."""
        assert extract_base_command("cd /tmp && ls") == "cd"


class TestCheckCommand:
    def test_allowed_command(self) -> None:
        allowed, reason = check_command("git status", ["git", "npm"], None)
        assert allowed
        assert reason == ""

    def test_blocked_command(self) -> None:
        allowed, reason = check_command("rm -rf /", None, ["rm", "dd"])
        assert not allowed
        assert "blocked" in reason

    def test_blocklist_precedence(self) -> None:
        """Blocklist takes precedence over allowlist."""
        allowed, _reason = check_command("rm file.txt", ["rm"], ["rm"])
        assert not allowed

    def test_no_lists_allows_all(self) -> None:
        allowed, _reason = check_command("anything", None, None)
        assert allowed

    def test_not_in_allowlist(self) -> None:
        allowed, reason = check_command("curl http://x", ["git", "npm"], None)
        assert not allowed
        assert "not in allowed list" in reason

    def test_empty_command(self) -> None:
        allowed, reason = check_command("", None, None)
        assert not allowed
        assert "Empty command" in reason

    def test_empty_allowlist(self) -> None:
        """Empty allowlist [] is falsy, so treated as 'no restriction'."""
        allowed, _ = check_command("git status", [], None)
        assert allowed
