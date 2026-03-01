"""Domain extraction and allowlist matching for Network.Http capability."""

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


def check_domain(
    url: str,
    allowed_domains: list[str] | None,
) -> tuple[bool, str]:
    """Check whether a URL's domain is permitted by the policy.

    Args:
        url: The full URL to check.
        allowed_domains: Domain allowlist. If None, all domains allowed.

    Returns:
        (allowed, reason) tuple.
    """
    if not allowed_domains:
        return True, ""

    domain = extract_domain(url)
    if not domain:
        return False, f"Could not extract domain from URL: {url}"

    for allowed in allowed_domains:
        # Exact match or subdomain match
        if domain == allowed or domain.endswith("." + allowed):
            return True, ""

    return False, f"Domain not in allowed list: {domain}"
