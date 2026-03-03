"""Data models for code execution results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeExecutionResult:
    """Structured result from a Python script execution."""

    stdout: str
    stderr: str
    exit_code: int
    images: list[bytes] = field(default_factory=list)  # PNG data from output dir
    execution_time: float = 0.0  # seconds
    timed_out: bool = False
