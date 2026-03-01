"""Tests for tool_runtime.validation security validators and input validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from tool_runtime.exceptions import ToolInputValidationError
from tool_runtime.models import ExecutionContext, RawToolOutput
from tool_runtime.tools.base import BaseTool
from tool_runtime.validation import (
    is_private_ip,
    validate_absolute_path,
    validate_no_null_bytes,
    validate_url,
)


class TestValidateNoNullBytes:
    def test_clean_string_passes(self) -> None:
        validate_no_null_bytes("hello world")

    def test_null_byte_raises(self) -> None:
        with pytest.raises(ToolInputValidationError, match="null bytes"):
            validate_no_null_bytes("hello\x00world")

    def test_empty_string_passes(self) -> None:
        validate_no_null_bytes("")


class TestValidateAbsolutePath:
    def test_absolute_path_passes(self) -> None:
        validate_absolute_path("/home/user/file.txt")

    def test_relative_path_raises(self) -> None:
        with pytest.raises(ToolInputValidationError, match="absolute"):
            validate_absolute_path("relative/path.txt")

    def test_null_byte_in_path_raises(self) -> None:
        with pytest.raises(ToolInputValidationError, match="null bytes"):
            validate_absolute_path("/home/user\x00/etc/passwd")

    def test_dot_dot_path_still_passes(self) -> None:
        # Path traversal detection via symlink resolution is the platform adapter's job.
        # validate_absolute_path only checks absoluteness and null bytes.
        validate_absolute_path("/home/user/../etc/passwd")


class TestIsPrivateIp:
    def test_loopback_is_private(self) -> None:
        assert is_private_ip("127.0.0.1") is True

    def test_rfc1918_is_private(self) -> None:
        assert is_private_ip("192.168.1.1") is True
        assert is_private_ip("10.0.0.1") is True
        assert is_private_ip("172.16.0.1") is True

    def test_ipv6_loopback_is_private(self) -> None:
        assert is_private_ip("::1") is True

    def test_link_local_is_private(self) -> None:
        assert is_private_ip("169.254.1.1") is True

    def test_unresolvable_host_fails_closed(self) -> None:
        assert is_private_ip("this-host-definitely-does-not-exist-xyz.invalid") is True

    def test_public_ip_is_not_private(self) -> None:
        assert is_private_ip("8.8.8.8") is False

    @patch("tool_runtime.validation.socket.getaddrinfo")
    def test_hostname_resolving_to_public_ip(self, mock_getaddrinfo: object) -> None:
        """Hostname that resolves to a public IP is not private."""
        import socket

        mock_getaddrinfo.return_value = [  # type: ignore[attr-defined]
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
        ]
        assert is_private_ip("example.com") is False

    @patch("tool_runtime.validation.socket.getaddrinfo")
    def test_hostname_resolving_to_private_ip(self, mock_getaddrinfo: object) -> None:
        """Hostname that resolves to a private IP is private."""
        import socket

        mock_getaddrinfo.return_value = [  # type: ignore[attr-defined]
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
        ]
        assert is_private_ip("internal.corp") is True


class TestValidateUrl:
    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    def test_valid_https_url(self, _mock: object) -> None:
        validate_url("https://api.example.com/data")

    @patch("tool_runtime.validation.is_private_ip", return_value=False)
    def test_valid_http_url(self, _mock: object) -> None:
        validate_url("http://api.example.com/data")

    def test_ftp_scheme_rejected(self) -> None:
        with pytest.raises(ToolInputValidationError, match="scheme"):
            validate_url("ftp://files.example.com/data")

    def test_no_scheme_rejected(self) -> None:
        with pytest.raises(ToolInputValidationError, match="scheme"):
            validate_url("api.example.com/data")

    def test_no_hostname_rejected(self) -> None:
        with pytest.raises(ToolInputValidationError, match="hostname"):
            validate_url("http://")

    def test_null_byte_rejected(self) -> None:
        with pytest.raises(ToolInputValidationError, match="null bytes"):
            validate_url("https://example.com/\x00path")

    def test_localhost_rejected(self) -> None:
        with pytest.raises(ToolInputValidationError, match="private"):
            validate_url("http://127.0.0.1/admin")

    def test_private_ip_rejected(self) -> None:
        with pytest.raises(ToolInputValidationError, match="private"):
            validate_url("http://192.168.1.1/admin")


class _FakeTool(BaseTool):
    """Minimal tool for testing validate_input."""

    @property
    def name(self) -> str:
        return "FakeTool"

    @property
    def description(self) -> str:
        return "fake"

    @property
    def capability(self) -> str:
        return "Test.Fake"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 100},
                "mode": {"type": "string", "enum": ["fast", "slow"]},
                "verbose": {"type": "boolean"},
            },
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any], context: ExecutionContext) -> RawToolOutput:
        return RawToolOutput(output_text="ok")


class TestValidateInput:
    def setup_method(self) -> None:
        self.tool = _FakeTool()

    def test_valid_input_passes(self) -> None:
        self.tool.validate_input({"path": "/tmp/f", "count": 5, "mode": "fast"})

    def test_missing_required_field(self) -> None:
        with pytest.raises(ToolInputValidationError, match=r"Missing required.*path"):
            self.tool.validate_input({})

    def test_wrong_type_string(self) -> None:
        with pytest.raises(ToolInputValidationError, match="must be string"):
            self.tool.validate_input({"path": 123})

    def test_wrong_type_integer(self) -> None:
        with pytest.raises(ToolInputValidationError, match="must be integer"):
            self.tool.validate_input({"path": "/tmp/f", "count": "five"})

    def test_boolean_rejected_as_integer(self) -> None:
        with pytest.raises(ToolInputValidationError, match="must be integer"):
            self.tool.validate_input({"path": "/tmp/f", "count": True})

    def test_minimum_violated(self) -> None:
        with pytest.raises(ToolInputValidationError, match=">= 1"):
            self.tool.validate_input({"path": "/tmp/f", "count": 0})

    def test_maximum_violated(self) -> None:
        with pytest.raises(ToolInputValidationError, match="<= 100"):
            self.tool.validate_input({"path": "/tmp/f", "count": 101})

    def test_enum_violated(self) -> None:
        with pytest.raises(ToolInputValidationError, match="must be one of"):
            self.tool.validate_input({"path": "/tmp/f", "mode": "turbo"})

    def test_additional_properties_rejected(self) -> None:
        with pytest.raises(ToolInputValidationError, match="Unknown argument"):
            self.tool.validate_input({"path": "/tmp/f", "extra": "nope"})

    def test_wrong_type_boolean(self) -> None:
        with pytest.raises(ToolInputValidationError, match="must be boolean"):
            self.tool.validate_input({"path": "/tmp/f", "verbose": "yes"})
