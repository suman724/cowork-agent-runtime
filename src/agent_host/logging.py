"""Dual-output logging configuration for the Agent Host.

Configures structlog with two outputs:
  - stderr: Human-readable ConsoleRenderer (for dev/debugging)
  - file: Structured JSON with rotation (for support/troubleshooting)

Stdout is reserved for JSON-RPC messages — logging must never write to stdout.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping  # noqa: TC003
from logging.handlers import RotatingFileHandler
from pathlib import Path  # noqa: TC003 — used at runtime in function signatures
from typing import Any

import structlog

# Keys whose values are replaced with [REDACTED] in log output.
_SENSITIVE_KEYS = frozenset({"token", "auth", "password", "secret", "credential", "api_key"})

# 10 MB per file, 5 backups → ~60 MB max disk usage
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def _redact_sensitive_keys(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Replace values for sensitive keys with [REDACTED]."""
    for key in event_dict:
        if any(s in key.lower() for s in _SENSITIVE_KEYS):
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(log_level: str, log_dir: Path) -> None:
    """Set up dual-output structured logging.

    Args:
        log_level: Minimum log level (e.g. "info", "debug").
        log_dir: Directory for rotated JSON log files.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Shared pre-chain processors (run before formatting)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _redact_sensitive_keys,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # Configure structlog to use stdlib integration
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper()),
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # --- stderr handler (human-readable) ---
    stderr_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(),
    )
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(stderr_formatter)

    # --- file handler (JSON, rotated) ---
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )
    file_handler = RotatingFileHandler(
        log_dir / "agent-host.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)

    # Attach both handlers to the root logger
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stderr_handler)
    root.addHandler(file_handler)
    root.setLevel(logging.getLevelName(log_level.upper()))

    # Suppress noisy third-party loggers that write to stdout.
    # LiteLLM adds its own StreamHandler(stdout) and emits verbose debug
    # output that would corrupt the JSON-RPC channel on stdout.
    for name in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router", "httpx", "httpcore"):
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(logging.WARNING)
        lib_logger.handlers.clear()
        lib_logger.propagate = True
