"""Tool runtime exception hierarchy.

These exceptions are internal to tool_runtime — the ToolRouter catches them
and maps to ToolResult(status="failed") with appropriate error codes.
"""

from __future__ import annotations


class ToolRuntimeError(Exception):
    """Base exception for all tool runtime errors."""

    code: str = "TOOL_EXECUTION_FAILED"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ToolNotFoundError(ToolRuntimeError):
    """Raised when a requested tool name is not registered."""

    code = "TOOL_NOT_FOUND"


class ToolInputValidationError(ToolRuntimeError):
    """Raised when tool input arguments fail validation."""

    code = "INVALID_REQUEST"


class FileNotFoundToolError(ToolRuntimeError):
    """Raised when a file operation targets a non-existent path."""

    code = "FILE_NOT_FOUND"


class FileTooLargeError(ToolRuntimeError):
    """Raised when a file exceeds the maximum allowed size."""

    code = "FILE_TOO_LARGE"


class ToolExecutionError(ToolRuntimeError):
    """Raised when a tool fails during execution."""

    code = "TOOL_EXECUTION_FAILED"


class ToolTimeoutError(ToolRuntimeError):
    """Raised when a tool execution exceeds the timeout."""

    code = "TOOL_EXECUTION_TIMEOUT"


class ToolPermissionError(ToolRuntimeError):
    """Raised when a file operation is denied by the OS."""

    code = "PERMISSION_DENIED"


# --- Browser automation exceptions ---


class BrowserLaunchError(ToolRuntimeError):
    """Raised when Playwright fails to start the browser."""

    code = "BROWSER_LAUNCH_FAILED"


class BrowserDomainBlockedError(ToolRuntimeError):
    """Raised when navigation targets a domain in blockedDomains."""

    code = "BROWSER_DOMAIN_BLOCKED"


class BrowserDomainDeniedError(ToolRuntimeError):
    """Raised when user denies domain approval."""

    code = "BROWSER_DOMAIN_DENIED"


class BrowserNavigationError(ToolRuntimeError):
    """Raised when page navigation fails (network error, invalid URL)."""

    code = "BROWSER_NAVIGATION_FAILED"


class BrowserAuthRequiredError(ToolRuntimeError):
    """Raised when a login page is detected — user takeover needed."""

    code = "BROWSER_AUTH_REQUIRED"


class BrowserElementNotFoundError(ToolRuntimeError):
    """Raised when an element index doesn't match any interactive element."""

    code = "BROWSER_ELEMENT_NOT_FOUND"


class BrowserElementNotInteractableError(ToolRuntimeError):
    """Raised when an element is hidden, disabled, or covered."""

    code = "BROWSER_ELEMENT_NOT_INTERACTABLE"


class BrowserWaitTimeoutError(ToolRuntimeError):
    """Raised when a wait condition is not met within the timeout."""

    code = "BROWSER_WAIT_TIMEOUT"


class BrowserDownloadError(ToolRuntimeError):
    """Raised when a file download fails."""

    code = "BROWSER_DOWNLOAD_FAILED"


class BrowserDownloadTimeoutError(ToolRuntimeError):
    """Raised when a download exceeds the timeout."""

    code = "BROWSER_DOWNLOAD_TIMEOUT"


class BrowserPathDeniedError(ToolRuntimeError):
    """Raised when a download path is outside allowed directories."""

    code = "BROWSER_PATH_DENIED"


class BrowserSensitiveDeniedError(ToolRuntimeError):
    """Raised when user denies a sensitive action approval."""

    code = "BROWSER_SENSITIVE_DENIED"


class BrowserSubmitDeniedError(ToolRuntimeError):
    """Raised when user denies a form submission approval."""

    code = "BROWSER_SUBMIT_DENIED"


class BrowserCrashedError(ToolRuntimeError):
    """Raised when the browser process crashes unexpectedly."""

    code = "BROWSER_CRASHED"


class BrowserDomainApprovalRequiredError(ToolRuntimeError):
    """Raised when navigation to a new domain requires user approval.

    ToolExecutor catches this and routes through ApprovalGate.
    Not a user-facing error — it's an internal signal.
    """

    code = "APPROVAL_REQUIRED"

    def __init__(self, domain: str) -> None:
        self.domain = domain
        super().__init__(f"Domain approval required: {domain}")


class BrowserSensitiveApprovalRequiredError(ToolRuntimeError):
    """Raised when a sensitive element interaction requires user approval.

    ToolExecutor catches this and routes through ApprovalGate.
    """

    code = "APPROVAL_REQUIRED"

    def __init__(self, action_summary: str, screenshot_base64: str | None = None) -> None:
        self.action_summary = action_summary
        self.screenshot_base64 = screenshot_base64
        super().__init__(f"Sensitive action approval required: {action_summary}")


class BrowserSubmitApprovalRequiredError(ToolRuntimeError):
    """Raised when a form submission requires user approval.

    ToolExecutor catches this and routes through ApprovalGate.
    Always raised by BrowserSubmit — unconditional.
    """

    code = "APPROVAL_REQUIRED"

    def __init__(
        self,
        description: str,
        url: str,
        form_data: list[dict[str, str]],
        screenshot_base64: str | None = None,
    ) -> None:
        self.description = description
        self.url = url
        self.form_data = form_data
        self.screenshot_base64 = screenshot_base64
        super().__init__(f"Form submission approval required: {description}")
