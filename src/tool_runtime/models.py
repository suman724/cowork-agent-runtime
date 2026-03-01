"""Tool runtime data models.

These are internal models used within tool_runtime. They complement (but do not
duplicate) the generated contract models from cowork-platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cowork_platform.tool_result import ToolResult

# --- Constants ---

DEFAULT_MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
DEFAULT_MAX_OUTPUT_BYTES: int = 100 * 1024  # 100 KB
ARTIFACT_THRESHOLD_BYTES: int = 10 * 1024  # 10 KB
DEFAULT_COMMAND_TIMEOUT_SECONDS: int = 300  # 5 minutes
DEFAULT_HTTP_TIMEOUT_SECONDS: int = 30
DEFAULT_MAX_RESPONSE_BYTES: int = 10 * 1024 * 1024  # 10 MB


@dataclass(frozen=True)
class ExecutionContext:
    """Carries optional capability constraints from agent_host.

    When fields are None, tools apply sensible defaults. This keeps
    tool_runtime testable in isolation without requiring a policy bundle.
    """

    allowed_paths: list[str] | None = None
    blocked_paths: list[str] | None = None
    allowed_commands: list[str] | None = None
    blocked_commands: list[str] | None = None
    allowed_domains: list[str] | None = None
    max_file_size_bytes: int | None = None
    max_output_bytes: int | None = None
    command_timeout_seconds: int | None = None
    working_directory: str | None = None


@dataclass
class ArtifactData:
    """Raw artifact bytes extracted from tool output."""

    artifact_type: str
    artifact_name: str
    data: bytes
    media_type: str = "text/plain"


@dataclass
class RawToolOutput:
    """Internal output from a tool before being wrapped in ToolResult.

    If the output exceeds ARTIFACT_THRESHOLD, the full output is extracted
    into artifact_data and output_text contains the truncated version.
    """

    output_text: str
    artifact_data: ArtifactData | None = None


@dataclass
class ToolExecutionResult:
    """Wraps ToolResult with optional artifact data.

    ToolResult is a generated Pydantic model with extra="forbid", so it
    cannot carry raw artifact bytes. This wrapper holds both.
    """

    tool_result: ToolResult
    artifacts: list[ArtifactData] = field(default_factory=list)
