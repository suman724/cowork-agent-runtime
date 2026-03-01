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
