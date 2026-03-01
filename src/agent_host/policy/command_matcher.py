"""Shell command matching — extracts base command and checks allow/block lists."""

from __future__ import annotations

import shlex


def extract_base_command(command: str) -> str:
    """Extract the base command name from a shell command string.

    Examples:
        "pytest tests/ --verbose" -> "pytest"
        "python -m build" -> "python"
        "/usr/bin/git status" -> "git"
        "cd /tmp && ls -la" -> "cd"
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        # Malformed quoting — fall back to whitespace split
        tokens = command.split()

    if not tokens:
        return ""

    first = tokens[0]
    # Strip path prefix: /usr/bin/git -> git
    if "/" in first:
        first = first.rsplit("/", maxsplit=1)[-1]
    if "\\" in first:
        first = first.rsplit("\\", maxsplit=1)[-1]

    return first


def check_command(
    command: str,
    allowed_commands: list[str] | None,
    blocked_commands: list[str] | None,
) -> tuple[bool, str]:
    """Check whether a shell command is permitted by the policy.

    Args:
        command: The full command string.
        allowed_commands: Base command allowlist. If None, all commands allowed.
        blocked_commands: Base command blocklist. Takes precedence.

    Returns:
        (allowed, reason) tuple.
    """
    base = extract_base_command(command)

    if not base:
        return False, "Empty command"

    # Blocklist takes precedence
    if blocked_commands and base in blocked_commands:
        return False, f"Command blocked: {base}"

    # If no allowlist, everything not blocked is allowed
    if not allowed_commands:
        return True, ""

    if base in allowed_commands:
        return True, ""

    return False, f"Command not in allowed list: {base}"
