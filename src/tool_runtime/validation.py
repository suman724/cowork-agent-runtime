"""Defense-in-depth security validators.

These validators provide baseline security checks regardless of policy.
Policy enforcement (allowed_paths, allowed_commands) is agent_host's job.
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import PurePath
from urllib.parse import urlparse

from tool_runtime.exceptions import ToolInputValidationError


def validate_no_null_bytes(value: str, field_name: str = "input") -> None:
    """Reject strings containing null bytes (path traversal / injection vector)."""
    if "\x00" in value:
        raise ToolInputValidationError(f"{field_name} must not contain null bytes")


def validate_absolute_path(path: str) -> None:
    """Validate that a path is absolute and contains no null bytes.

    Does NOT resolve symlinks — that's the caller's responsibility via
    the platform adapter.
    """
    validate_no_null_bytes(path, "path")
    if not PurePath(path).is_absolute():
        raise ToolInputValidationError(f"Path must be absolute, got: {path}")


def is_private_ip(hostname: str) -> bool:
    """Check whether a hostname resolves to a private/reserved IP address."""
    try:
        addr = ipaddress.ip_address(hostname)
        return addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local
    except ValueError:
        pass

    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canonname, sockaddr in results:
            ip_str = sockaddr[0]
            addr = ipaddress.ip_address(ip_str)
            if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local:
                return True
    except (socket.gaierror, OSError):
        return True  # fail-closed: unresolvable hosts are treated as private

    return False


def validate_url(url: str) -> None:
    """Validate a URL for HTTP requests with SSRF prevention.

    Checks:
    - Valid URL structure with http or https scheme
    - No null bytes
    - Hostname does not resolve to private/loopback/reserved IP
    """
    validate_no_null_bytes(url, "url")

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ToolInputValidationError(
            f"URL scheme must be http or https, got: {parsed.scheme or '(none)'}"
        )

    if not parsed.hostname:
        raise ToolInputValidationError("URL must have a hostname")

    if is_private_ip(parsed.hostname):
        raise ToolInputValidationError(
            f"URL resolves to a private/reserved IP address: {parsed.hostname}"
        )
