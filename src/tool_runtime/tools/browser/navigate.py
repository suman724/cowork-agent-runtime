"""BrowserNavigate — navigate the browser to a URL."""

from __future__ import annotations

import ipaddress
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import structlog

from tool_runtime.exceptions import (
    BrowserAuthRequiredError,
    BrowserDomainApprovalRequiredError,
    BrowserDomainBlockedError,
    BrowserNavigationError,
)
from tool_runtime.tools.browser.base_browser_tool import BaseBrowserTool

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

logger = structlog.get_logger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}

# Private/local IP ranges to block (SSRF prevention)
_LOCAL_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
]

# Auth detection signals
_AUTH_URL_PATTERNS = re.compile(
    r"/(login|signin|sign-in|auth|sso|oauth|authenticate)", re.IGNORECASE
)


class BrowserNavigateTool(BaseBrowserTool):
    """Navigate the browser to a URL."""

    @property
    def name(self) -> str:
        return "BrowserNavigate"

    @property
    def description(self) -> str:
        return (
            "Navigate the browser to a URL. Returns the page state with "
            "indexed interactive elements. Only http:// and https:// URLs allowed."
        )

    @property
    def capability(self) -> str:
        return "Browser.Navigate"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to (http:// or https://).",
                },
                "waitUntil": {
                    "type": "string",
                    "enum": ["domcontentloaded", "load", "networkidle"],
                    "description": (
                        "Wait condition. 'domcontentloaded' (default, fastest), "
                        "'load' (all resources), 'networkidle' (no requests for 500ms)."
                    ),
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> RawToolOutput:
        self.validate_input(arguments)
        url: str = arguments["url"]
        wait_until: str = arguments.get("waitUntil", "domcontentloaded")

        # 1. URL validation
        self._validate_url(url)

        # 2. SSRF prevention
        self._check_ssrf(url)

        # 3. Domain policy check
        domain = urlparse(url).hostname or ""
        self._check_domain_policy(domain, context)

        # 4. Domain approval (Tier 1)
        self._check_domain_approval(domain, context)

        # 5. Navigate
        page = await self._get_page()
        try:
            response = await page.goto(url, wait_until=wait_until)  # type: ignore[arg-type]
        except Exception as exc:
            raise BrowserNavigationError(f"Navigation failed: {exc}") from exc

        # 6. Check for auth page
        if response and response.status in (401, 403):
            raise BrowserAuthRequiredError(
                f"Authentication required (HTTP {response.status}) for {domain}"
            )

        # 7. Auth detection via URL/content heuristics
        current_url = page.url
        if _AUTH_URL_PATTERNS.search(current_url):
            snapshot = await self._extract_snapshot(page)
            has_password = any(el.raw.get("inputType") == "password" for el in snapshot.elements)
            if has_password:
                raise BrowserAuthRequiredError(
                    f"Login page detected on {domain}. "
                    "Please log in using the browser window, then click Resume."
                )

        # 8. Extract page state
        rendered = await self._extract_and_render(page)

        logger.info("browser_navigated", url=url, domain=domain)
        return self._page_state_output(rendered)

    def _validate_url(self, url: str) -> None:
        """Ensure URL uses http:// or https:// scheme."""
        parsed = urlparse(url)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            raise BrowserNavigationError(
                f"Only http:// and https:// URLs are allowed, got {parsed.scheme}://"
            )

    def _check_ssrf(self, url: str) -> None:
        """Block navigation to local/private IP addresses."""
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        try:
            addr = ipaddress.ip_address(hostname)
            for network in _LOCAL_NETWORKS:
                if addr in network:
                    raise BrowserNavigationError(
                        f"Navigation to local/private address {hostname} is blocked"
                    )
        except ValueError:
            pass  # Not an IP address — hostname is fine

    def _check_domain_policy(self, domain: str, context: ExecutionContext) -> None:
        """Check domain against policy allowedDomains/blockedDomains."""
        # Blocked domains take precedence
        if context.blocked_domains:
            for pattern in context.blocked_domains:
                if _domain_matches(domain, pattern):
                    raise BrowserDomainBlockedError(f"Domain {domain} is blocked by policy")

        # If allowedDomains is set, domain must match at least one
        if context.allowed_domains:
            for pattern in context.allowed_domains:
                if _domain_matches(domain, pattern):
                    return  # Allowed
            raise BrowserDomainBlockedError(f"Domain {domain} is not in the allowed domains list")

    def _check_domain_approval(self, domain: str, context: ExecutionContext) -> None:
        """Check if domain is pre-approved or needs user approval (Tier 1)."""
        # Pre-approved via policy allowedDomains
        if context.allowed_domains:
            for pattern in context.allowed_domains:
                if _domain_matches(domain, pattern):
                    return  # Pre-approved

        # Session-approved (already visited and approved)
        if domain in self._browser_manager.approved_domains:
            return

        # Needs approval — raise signal for ToolExecutor to handle
        raise BrowserDomainApprovalRequiredError(domain)


def _domain_matches(domain: str, pattern: str) -> bool:
    """Match a domain against a policy pattern.

    Supports wildcard prefix: "*.example.com" matches "sub.example.com".
    Exact match: "example.com" matches only "example.com".
    """
    if pattern.startswith("*."):
        suffix = pattern[1:]  # ".example.com"
        return domain == pattern[2:] or domain.endswith(suffix)
    return domain == pattern
