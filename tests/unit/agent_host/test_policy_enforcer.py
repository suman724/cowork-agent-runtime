"""Tests for the PolicyEnforcer — security boundary, extensive coverage."""

from __future__ import annotations

from pathlib import Path

from cowork_platform_sdk import CapabilityName

from agent_host.policy.policy_enforcer import PolicyEnforcer
from tests.fixtures.policy_bundles import make_policy_bundle, make_restrictive_bundle

# ---- Expiry ----


class TestPolicyExpiry:
    def test_not_expired(self) -> None:
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        assert not enforcer.is_expired()

    def test_expired(self) -> None:
        bundle = make_policy_bundle(expired=True)
        enforcer = PolicyEnforcer(bundle)
        assert enforcer.is_expired()

    def test_expired_denies_tool_call(self) -> None:
        bundle = make_policy_bundle(expired=True)
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call("ReadFile", CapabilityName.FILE_READ, {"path": "/tmp/x"})
        assert result.decision == "DENIED"
        assert "expired" in result.reason.lower()

    def test_expired_denies_llm_call(self) -> None:
        bundle = make_policy_bundle(expired=True)
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_llm_call()
        assert result.decision == "DENIED"
        assert "expired" in result.reason.lower()


# ---- Capability granted ----


class TestCapabilityGrant:
    def test_all_capabilities_granted(self) -> None:
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call("ReadFile", CapabilityName.FILE_READ, {"path": "/tmp/x"})
        assert result.decision == "ALLOWED"

    def test_capability_not_granted(self) -> None:
        bundle = make_policy_bundle(capabilities=[{"name": "File.Read"}, {"name": "LLM.Call"}])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "WriteFile", CapabilityName.FILE_WRITE, {"path": "/tmp/x", "content": "hi"}
        )
        assert result.decision == "DENIED"
        assert "not granted" in result.reason.lower()

    def test_llm_call_not_granted(self) -> None:
        bundle = make_policy_bundle(capabilities=[{"name": "File.Read"}])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_llm_call()
        assert result.decision == "DENIED"
        assert "not granted" in result.reason.lower()

    def test_llm_call_granted(self) -> None:
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_llm_call()
        assert result.decision == "ALLOWED"


# ---- Path enforcement ----


class TestPathEnforcement:
    def test_allowed_path(self, tmp_path: object) -> None:
        path = str(tmp_path)
        bundle = make_restrictive_bundle(allowed_paths=[path])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call("ReadFile", CapabilityName.FILE_READ, {"path": path})
        assert result.decision == "ALLOWED"

    def test_path_under_allowed_prefix(self, tmp_path: Path) -> None:
        path = str(tmp_path)
        sub = str(tmp_path / "subdir" / "file.txt")
        bundle = make_restrictive_bundle(allowed_paths=[path])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call("ReadFile", CapabilityName.FILE_READ, {"path": sub})
        assert result.decision == "ALLOWED"

    def test_path_not_allowed(self, tmp_path: object) -> None:
        allowed = str(tmp_path)
        bundle = make_restrictive_bundle(allowed_paths=[allowed])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "ReadFile", CapabilityName.FILE_READ, {"path": "/etc/passwd"}
        )
        assert result.decision == "DENIED"
        assert "not in allowed" in result.reason.lower()

    def test_blocked_path_overrides_allowed(self, tmp_path: Path) -> None:
        path = str(tmp_path)
        secret = str(tmp_path / "secret")
        bundle = make_restrictive_bundle(allowed_paths=[path], blocked_paths=[secret])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "ReadFile",
            CapabilityName.FILE_READ,
            {"path": str(Path(secret) / "key.pem")},
        )
        assert result.decision == "DENIED"
        assert "blocked" in result.reason.lower()

    def test_no_path_constraints_allows_all(self) -> None:
        bundle = make_restrictive_bundle()
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "ReadFile", CapabilityName.FILE_READ, {"path": "/any/path"}
        )
        assert result.decision == "ALLOWED"

    def test_write_path_enforcement(self, tmp_path: object) -> None:
        allowed = str(tmp_path)
        bundle = make_restrictive_bundle(allowed_paths=[allowed])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "WriteFile",
            CapabilityName.FILE_WRITE,
            {"path": "/etc/danger", "content": "x"},
        )
        assert result.decision == "DENIED"

    def test_delete_path_enforcement(self, tmp_path: object) -> None:
        allowed = str(tmp_path)
        bundle = make_restrictive_bundle(allowed_paths=[allowed])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "DeleteFile", CapabilityName.FILE_DELETE, {"path": "/etc/danger"}
        )
        assert result.decision == "DENIED"

    def test_symlink_resolved(self, tmp_path: object) -> None:
        """Symlinks are resolved before path matching."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real_dir)

        bundle = make_restrictive_bundle(allowed_paths=[str(real_dir)])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "ReadFile",
            CapabilityName.FILE_READ,
            {"path": str(link / "file.txt")},
        )
        assert result.decision == "ALLOWED"


# ---- Command enforcement ----


class TestCommandEnforcement:
    def test_allowed_command(self) -> None:
        bundle = make_restrictive_bundle(allowed_commands=["pytest", "python", "git"])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "RunCommand", CapabilityName.SHELL_EXEC, {"command": "pytest tests/ --verbose"}
        )
        assert result.decision == "ALLOWED"

    def test_command_not_allowed(self) -> None:
        bundle = make_restrictive_bundle(allowed_commands=["pytest", "python"])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "RunCommand", CapabilityName.SHELL_EXEC, {"command": "rm -rf /"}
        )
        assert result.decision == "DENIED"
        assert "not in allowed" in result.reason.lower()

    def test_blocked_command(self) -> None:
        bundle = make_restrictive_bundle(blocked_commands=["rm", "sudo"])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "RunCommand", CapabilityName.SHELL_EXEC, {"command": "rm -rf /tmp"}
        )
        assert result.decision == "DENIED"
        assert "blocked" in result.reason.lower()

    def test_blocked_overrides_allowed(self) -> None:
        bundle = make_restrictive_bundle(allowed_commands=["git", "rm"], blocked_commands=["rm"])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "RunCommand", CapabilityName.SHELL_EXEC, {"command": "rm -rf /"}
        )
        assert result.decision == "DENIED"

    def test_no_command_constraints_allows_all(self) -> None:
        bundle = make_restrictive_bundle()
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "RunCommand", CapabilityName.SHELL_EXEC, {"command": "anything --flag"}
        )
        assert result.decision == "ALLOWED"

    def test_path_stripped_from_command(self) -> None:
        bundle = make_restrictive_bundle(allowed_commands=["git"])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "RunCommand", CapabilityName.SHELL_EXEC, {"command": "/usr/bin/git status"}
        )
        assert result.decision == "ALLOWED"

    def test_empty_command_denied(self) -> None:
        bundle = make_restrictive_bundle(allowed_commands=["git"])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call("RunCommand", CapabilityName.SHELL_EXEC, {"command": ""})
        assert result.decision == "DENIED"


# ---- Domain enforcement ----


class TestDomainEnforcement:
    def test_allowed_domain(self) -> None:
        bundle = make_restrictive_bundle(allowed_domains=["api.example.com"])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "HttpRequest",
            CapabilityName.NETWORK_HTTP,
            {"url": "https://api.example.com/v1/data"},
        )
        assert result.decision == "ALLOWED"

    def test_subdomain_match(self) -> None:
        bundle = make_restrictive_bundle(allowed_domains=["example.com"])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "HttpRequest",
            CapabilityName.NETWORK_HTTP,
            {"url": "https://api.example.com/v1/data"},
        )
        assert result.decision == "ALLOWED"

    def test_domain_not_allowed(self) -> None:
        bundle = make_restrictive_bundle(allowed_domains=["example.com"])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "HttpRequest",
            CapabilityName.NETWORK_HTTP,
            {"url": "https://evil.com/steal"},
        )
        assert result.decision == "DENIED"
        assert "not in allowed" in result.reason.lower()

    def test_no_domain_constraints_allows_all(self) -> None:
        bundle = make_restrictive_bundle()
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "HttpRequest",
            CapabilityName.NETWORK_HTTP,
            {"url": "https://anything.com/data"},
        )
        assert result.decision == "ALLOWED"

    def test_invalid_url_domain(self) -> None:
        bundle = make_restrictive_bundle(allowed_domains=["example.com"])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "HttpRequest",
            CapabilityName.NETWORK_HTTP,
            {"url": "not-a-url"},
        )
        assert result.decision == "DENIED"


# ---- Approval required ----


class TestApprovalRequired:
    def test_approval_required(self) -> None:
        bundle = make_restrictive_bundle(requires_approval=True, approval_rule_id="rule-42")
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call("ReadFile", CapabilityName.FILE_READ, {"path": "/tmp/x"})
        assert result.decision == "APPROVAL_REQUIRED"
        assert result.approval_rule_id == "rule-42"
        assert result.risk_level is not None

    def test_approval_has_risk_level(self) -> None:
        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call("ReadFile", CapabilityName.FILE_READ, {"path": "/tmp/x"})
        assert result.risk_level == "low"  # File.Read is low risk


# ---- Risk assessment ----


class TestRiskAssessment:
    def test_file_read_low(self) -> None:
        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call("ReadFile", CapabilityName.FILE_READ, {"path": "/tmp/x"})
        assert result.risk_level == "low"

    def test_file_write_medium(self) -> None:
        bundle = make_restrictive_bundle(requires_approval=True)
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "WriteFile", CapabilityName.FILE_WRITE, {"path": "/tmp/x", "content": "hi"}
        )
        assert result.risk_level == "medium"

    def test_file_delete_high(self) -> None:
        # File.Delete doesn't have requiresApproval set in make_restrictive_bundle
        capabilities = [
            {
                "name": "File.Delete",
                "requiresApproval": True,
                "approvalRuleId": "rule-1",
            },
            {"name": "LLM.Call"},
        ]
        bundle = make_policy_bundle(
            capabilities=capabilities,
            approval_rules=[{"approvalRuleId": "rule-1", "title": "Test", "description": "Test"}],
        )
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "DeleteFile", CapabilityName.FILE_DELETE, {"path": "/tmp/x"}
        )
        assert result.risk_level == "high"


# ---- Edge cases ----


class TestEdgeCases:
    def test_policy_bundle_property(self) -> None:
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        assert enforcer.policy_bundle is bundle

    def test_empty_arguments(self) -> None:
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call("ReadFile", CapabilityName.FILE_READ, {})
        assert result.decision == "ALLOWED"

    def test_unknown_capability(self) -> None:
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call("CustomTool", "Custom.Thing", {"arg": "val"})
        assert result.decision == "DENIED"
        assert "not granted" in result.reason.lower()
