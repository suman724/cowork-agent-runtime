"""Domain matching for Network.Http capability — blocklist and allowlist."""

from __future__ import annotations

from urllib.parse import urlparse


def extract_domain(url: str) -> str:
    """Extract the domain (hostname) from a URL.

    Examples:
        "https://api.example.com/v1/data" -> "api.example.com"
        "http://localhost:8080/test" -> "localhost"
    """
    parsed = urlparse(url)
    return parsed.hostname or ""


def _domain_matches(domain: str, pattern: str) -> bool:
    """Check if a domain matches a pattern (exact or subdomain match)."""
    return domain == pattern or domain.endswith("." + pattern)


def check_domain(
    url: str,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None = None,
) -> tuple[bool, str]:
    """Check whether a URL's domain is permitted by the policy.

    Blocklist is checked first and takes precedence over the allowlist.

    Args:
        url: The full URL to check.
        allowed_domains: Domain allowlist. If None/empty, all domains allowed
            (unless blocked).
        blocked_domains: Domain blocklist. If a domain matches, it is denied
            regardless of the allowlist.

    Returns:
        (allowed, reason) tuple.
    """
    domain = extract_domain(url)
    if not domain:
        if allowed_domains or blocked_domains:
            return False, f"Could not extract domain from URL: {url}"
        return True, ""

    # 1. Blocklist takes precedence
    if blocked_domains:
        for blocked in blocked_domains:
            if _domain_matches(domain, blocked):
                return False, f"Domain blocked: {domain}"

    # 2. If no allowlist, everything not blocked is allowed
    if not allowed_domains:
        return True, ""

    # 3. Check allowlist
    for allowed in allowed_domains:
        if _domain_matches(domain, allowed):
            return True, ""

    return False, f"Domain not in allowed list: {domain}"
