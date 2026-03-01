"""Path prefix matching with blocklist precedence and symlink resolution."""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: str) -> str:
    """Resolve a path to its real absolute form (following symlinks)."""
    return str(Path(path).resolve())


def check_path(
    path: str,
    allowed_paths: list[str] | None,
    blocked_paths: list[str] | None,
) -> tuple[bool, str]:
    """Check whether a path is permitted by the policy.

    Args:
        path: The resolved, absolute path to check.
        allowed_paths: Prefix allowlist. If None, all paths allowed.
        blocked_paths: Prefix blocklist. Takes precedence over allowlist.

    Returns:
        (allowed, reason) tuple.
    """
    resolved = resolve_path(path)

    # Blocklist takes precedence
    if blocked_paths:
        for blocked in blocked_paths:
            resolved_blocked = resolve_path(blocked)
            if resolved == resolved_blocked or resolved.startswith(resolved_blocked + "/"):
                return False, f"Path blocked: {resolved} matches blocked prefix {resolved_blocked}"

    # If no allowlist, everything not blocked is allowed
    if not allowed_paths:
        return True, ""

    # Check allowlist
    for allowed in allowed_paths:
        resolved_allowed = resolve_path(allowed)
        if resolved == resolved_allowed or resolved.startswith(resolved_allowed + "/"):
            return True, ""

    return False, f"Path not in allowed list: {resolved}"
