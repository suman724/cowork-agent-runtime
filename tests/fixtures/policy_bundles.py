"""Factory functions for creating PolicyBundle test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from cowork_platform.policy_bundle import ApprovalRule, Capability, LlmPolicy, PolicyBundle


def make_policy_bundle(
    *,
    capabilities: list[dict[str, Any]] | None = None,
    expires_in_seconds: int = 3600,
    expired: bool = False,
    session_id: str = "sess-test",
    tenant_id: str = "tenant-test",
    user_id: str = "user-test",
    max_session_tokens: int = 100000,
    approval_rules: list[dict[str, Any]] | None = None,
) -> PolicyBundle:
    """Create a PolicyBundle for testing.

    Args:
        capabilities: List of capability dicts. Defaults to all capabilities granted.
        expires_in_seconds: Seconds until expiry (ignored if expired=True).
        expired: If True, creates an already-expired bundle.
        session_id: Session ID.
        tenant_id: Tenant ID.
        user_id: User ID.
        max_session_tokens: Max session token budget.
        approval_rules: List of approval rule dicts.
    """
    if expired:
        expires_at = datetime.now(tz=UTC) - timedelta(seconds=60)
    else:
        expires_at = datetime.now(tz=UTC) + timedelta(seconds=expires_in_seconds)

    if capabilities is None:
        capabilities = [
            {"name": "File.Read"},
            {"name": "File.Write"},
            {"name": "File.Delete"},
            {"name": "Shell.Exec"},
            {"name": "Network.Http"},
            {"name": "Workspace.Upload"},
            {"name": "LLM.Call"},
            {"name": "Search.Web"},
        ]

    cap_objects = [Capability(**c) for c in capabilities]

    rules = []
    if approval_rules:
        rules = [ApprovalRule(**r) for r in approval_rules]

    return PolicyBundle(
        policyBundleVersion="2026-02-28.1",
        schemaVersion="1.0",
        tenantId=tenant_id,
        userId=user_id,
        sessionId=session_id,
        expiresAt=expires_at,
        capabilities=cap_objects,
        llmPolicy=LlmPolicy(
            allowedModels=["gpt-4o"],
            maxInputTokens=8000,
            maxOutputTokens=4000,
            maxSessionTokens=max_session_tokens,
        ),
        approvalRules=rules,
    )


def make_restrictive_bundle(
    *,
    allowed_paths: list[str] | None = None,
    blocked_paths: list[str] | None = None,
    allowed_commands: list[str] | None = None,
    blocked_commands: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    requires_approval: bool = False,
    approval_rule_id: str = "rule-1",
) -> PolicyBundle:
    """Create a PolicyBundle with scope restrictions for testing."""
    capabilities: list[dict[str, Any]] = [
        {
            "name": "File.Read",
            "allowedPaths": allowed_paths,
            "blockedPaths": blocked_paths,
            "requiresApproval": requires_approval,
            "approvalRuleId": approval_rule_id if requires_approval else None,
        },
        {
            "name": "File.Write",
            "allowedPaths": allowed_paths,
            "blockedPaths": blocked_paths,
            "requiresApproval": requires_approval,
            "approvalRuleId": approval_rule_id if requires_approval else None,
        },
        {
            "name": "File.Delete",
            "allowedPaths": allowed_paths,
            "blockedPaths": blocked_paths,
        },
        {
            "name": "Shell.Exec",
            "allowedCommands": allowed_commands,
            "blockedCommands": blocked_commands,
        },
        {
            "name": "Network.Http",
            "allowedDomains": allowed_domains,
        },
        {"name": "LLM.Call"},
    ]

    approval_rules = []
    if requires_approval:
        approval_rules = [
            {
                "approvalRuleId": approval_rule_id,
                "title": "Test approval",
                "description": "Requires approval for testing",
            }
        ]

    return make_policy_bundle(
        capabilities=capabilities,
        approval_rules=approval_rules,
    )
