"""Risk level assessment for tool calls requiring approval."""

from __future__ import annotations

from cowork_platform_sdk import CapabilityName, RiskLevel


def assess_risk(
    tool_name: str,  # noqa: ARG001
    capability_name: str,
    arguments: dict[str, object],  # noqa: ARG001
) -> str:
    """Assess the risk level of a tool call.

    Args:
        tool_name: The tool name (reserved for future per-tool risk rules).
        capability_name: The capability being invoked.
        arguments: Tool call arguments (reserved for future argument-based risk).

    Returns:
        Risk level string: "low", "medium", or "high".
    """
    if capability_name == CapabilityName.FILE_READ:
        return RiskLevel.LOW

    if capability_name == CapabilityName.FILE_WRITE:
        return RiskLevel.MEDIUM

    if capability_name == CapabilityName.FILE_DELETE:
        return RiskLevel.HIGH

    if capability_name == CapabilityName.SHELL_EXEC:
        return RiskLevel.MEDIUM

    if capability_name == CapabilityName.NETWORK_HTTP:
        return RiskLevel.MEDIUM

    if capability_name == CapabilityName.WORKSPACE_UPLOAD:
        return RiskLevel.LOW

    if capability_name == CapabilityName.BACKEND_TOOL_INVOKE:
        return RiskLevel.MEDIUM

    # Unknown capability — default to high
    return RiskLevel.HIGH
