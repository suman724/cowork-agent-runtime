"""Tests for dual-output logging configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import structlog

from agent_host.logging import _redact_sensitive_keys, configure_logging


class TestConfigureLogging:
    def test_creates_log_directory(self, tmp_path: Path) -> None:
        """Creates the log directory if it does not exist."""
        log_dir = tmp_path / "nested" / "logs"
        assert not log_dir.exists()

        configure_logging("info", log_dir)

        assert log_dir.is_dir()

    def test_log_file_receives_json(self, tmp_path: Path) -> None:
        """Log entries are written as JSON lines to the file."""
        configure_logging("info", tmp_path)

        log = structlog.get_logger()
        log.info("test_event", key="value")

        log_file = tmp_path / "agent-host.log"
        assert log_file.exists()

        lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1

        record = json.loads(lines[-1])
        assert record["event"] == "test_event"
        assert record["key"] == "value"
        assert "timestamp" in record

    def test_stderr_handler_present(self, tmp_path: Path) -> None:
        """Root logger has a stderr StreamHandler attached."""
        configure_logging("debug", tmp_path)

        root = logging.getLogger()
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types
        assert "RotatingFileHandler" in handler_types


class TestRedactSensitiveKeys:
    def test_redacts_known_keys(self) -> None:
        """Keys containing sensitive substrings are replaced with [REDACTED]."""
        event_dict: dict[str, Any] = {
            "event": "login",
            "auth_token": "abc123",
            "api_key": "secret",
            "password": "hunter2",
            "user_id": "u-1",
        }

        result = _redact_sensitive_keys(None, "", event_dict)

        assert result["auth_token"] == "[REDACTED]"  # noqa: S105
        assert result["api_key"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"  # noqa: S105
        # Non-sensitive keys untouched
        assert result["event"] == "login"
        assert result["user_id"] == "u-1"

    def test_case_insensitive_match(self) -> None:
        """Sensitive key detection is case-insensitive."""
        event_dict: dict[str, Any] = {"LLM_AUTH_TOKEN": "tok-xyz"}

        result = _redact_sensitive_keys(None, "", event_dict)

        assert result["LLM_AUTH_TOKEN"] == "[REDACTED]"  # noqa: S105

    def test_no_sensitive_keys(self) -> None:
        """Non-sensitive event dicts pass through unchanged."""
        event_dict: dict[str, Any] = {"event": "ready", "port": 8080}

        result = _redact_sensitive_keys(None, "", event_dict)

        assert result == {"event": "ready", "port": 8080}
