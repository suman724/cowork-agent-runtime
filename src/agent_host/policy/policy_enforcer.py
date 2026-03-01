"""Policy Enforcer — validates tool calls against the session policy bundle.

Stateless and pure: receives PolicyBundle at init, indexes capabilities,
and checks tool calls without any I/O.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cowork_platform.policy_bundle import Capability, PolicyBundle  # noqa: TC002
from cowork_platform_sdk import CapabilityName

from agent_host.models import PolicyCheckResult
from agent_host.policy.command_matcher import check_command
from agent_host.policy.domain_matcher import check_domain
from agent_host.policy.path_matcher import check_path
from agent_host.policy.risk_assessor import assess_risk

# Capabilities that use path-based constraints
_PATH_CAPABILITIES = frozenset(
    {
        CapabilityName.FILE_READ,
        CapabilityName.FILE_WRITE,
        CapabilityName.FILE_DELETE,
    }
)


class PolicyEnforcer:
    """Validates tool calls against a policy bundle.

    Indexes capabilities by name at init for O(1) lookup. All methods
    are synchronous and side-effect-free.
    """

    def __init__(self, policy_bundle: PolicyBundle) -> None:
        self._policy_bundle = policy_bundle
        # Index capabilities by name for O(1) lookup
        self._capabilities: dict[str, Capability] = {
            cap.name: cap for cap in policy_bundle.capabilities
        }

    @property
    def policy_bundle(self) -> PolicyBundle:
        """Return the underlying policy bundle."""
        return self._policy_bundle

    def is_expired(self) -> bool:
        """Check if the policy bundle has expired."""
        return datetime.now(tz=UTC) >= self._policy_bundle.expiresAt

    def get_capability(self, capability_name: str) -> Capability | None:
        """Return the Capability for a given name, or None if not granted."""
        return self._capabilities.get(capability_name)

    def check_tool_call(
        self,
        tool_name: str,
        capability_name: str,
        arguments: dict[str, Any],
    ) -> PolicyCheckResult:
        """Check whether a tool call is permitted by the policy.

        Args:
            tool_name: The tool name (e.g. "ReadFile").
            capability_name: The capability required (e.g. "File.Read").
            arguments: The tool call arguments.

        Returns:
            PolicyCheckResult with decision ALLOWED, DENIED, or APPROVAL_REQUIRED.
        """
        # 1. Check expiry
        if self.is_expired():
            return PolicyCheckResult(
                decision="DENIED",
                reason="Policy bundle has expired",
            )

        # 2. Capability granted?
        capability = self._capabilities.get(capability_name)
        if capability is None:
            return PolicyCheckResult(
                decision="DENIED",
                reason=f"Capability not granted: {capability_name}",
            )

        # 3. Scope constraints
        deny = self._check_scope_constraints(capability_name, capability, arguments)
        if deny is not None:
            return deny

        # 4. Approval required?
        if capability.requiresApproval:
            risk_level = assess_risk(tool_name, capability_name, arguments)
            return PolicyCheckResult(
                decision="APPROVAL_REQUIRED",
                reason=f"Approval required for {capability_name}",
                approval_rule_id=capability.approvalRuleId or "",
                risk_level=risk_level,
            )

        return PolicyCheckResult(decision="ALLOWED")

    def check_llm_call(self) -> PolicyCheckResult:
        """Check whether an LLM call is permitted.

        Verifies LLM.Call capability is granted and policy is not expired.
        """
        if self.is_expired():
            return PolicyCheckResult(
                decision="DENIED",
                reason="Policy bundle has expired",
            )

        if CapabilityName.LLM_CALL not in self._capabilities:
            return PolicyCheckResult(
                decision="DENIED",
                reason="LLM.Call capability not granted",
            )

        return PolicyCheckResult(decision="ALLOWED")

    def _check_scope_constraints(
        self,
        capability_name: str,
        capability: Capability,
        arguments: dict[str, Any],
    ) -> PolicyCheckResult | None:
        """Check scope constraints for a capability. Returns DENIED result or None."""
        if capability_name in _PATH_CAPABILITIES:
            path = arguments.get("path", "")
            if path:
                allowed, reason = check_path(
                    path,
                    capability.allowedPaths,
                    capability.blockedPaths,
                )
                if not allowed:
                    return PolicyCheckResult(decision="DENIED", reason=reason)

        elif capability_name == CapabilityName.SHELL_EXEC:
            command = arguments.get("command", "")
            allowed, reason = check_command(
                command,
                capability.allowedCommands,
                capability.blockedCommands,
            )
            if not allowed:
                return PolicyCheckResult(decision="DENIED", reason=reason)

        elif capability_name == CapabilityName.NETWORK_HTTP:
            url = arguments.get("url", "")
            if url:
                allowed, reason = check_domain(url, capability.allowedDomains)
                if not allowed:
                    return PolicyCheckResult(decision="DENIED", reason=reason)

        return None
