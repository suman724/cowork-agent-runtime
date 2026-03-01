"""Tests for risk level assessment."""

from __future__ import annotations

from cowork_platform_sdk import CapabilityName, RiskLevel

from agent_host.policy.risk_assessor import assess_risk


class TestAssessRisk:
    def test_file_read_is_low(self) -> None:
        assert assess_risk("ReadFile", CapabilityName.FILE_READ, {}) == RiskLevel.LOW

    def test_file_write_is_medium(self) -> None:
        assert assess_risk("WriteFile", CapabilityName.FILE_WRITE, {}) == RiskLevel.MEDIUM

    def test_file_delete_is_high(self) -> None:
        assert assess_risk("DeleteFile", CapabilityName.FILE_DELETE, {}) == RiskLevel.HIGH

    def test_shell_exec_is_medium(self) -> None:
        assert assess_risk("RunCommand", CapabilityName.SHELL_EXEC, {}) == RiskLevel.MEDIUM

    def test_network_http_is_medium(self) -> None:
        assert assess_risk("HttpRequest", CapabilityName.NETWORK_HTTP, {}) == RiskLevel.MEDIUM

    def test_workspace_upload_is_low(self) -> None:
        assert assess_risk("Upload", CapabilityName.WORKSPACE_UPLOAD, {}) == RiskLevel.LOW

    def test_backend_tool_invoke_is_medium(self) -> None:
        assert (
            assess_risk("BackendTool", CapabilityName.BACKEND_TOOL_INVOKE, {}) == RiskLevel.MEDIUM
        )

    def test_unknown_capability_is_high(self) -> None:
        assert assess_risk("Unknown", "SomeUnknown.Capability", {}) == RiskLevel.HIGH
