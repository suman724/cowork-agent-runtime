"""Tests for Code.Execute policy enforcement."""

from __future__ import annotations

from cowork_platform_sdk import CapabilityName

from agent_host.policy.policy_enforcer import PolicyEnforcer
from tests.fixtures.policy_bundles import make_policy_bundle


class TestCodeExecutePolicy:
    def test_code_execute_allowed_by_default(self) -> None:
        bundle = make_policy_bundle()
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "ExecuteCode", CapabilityName.CODE_EXECUTE, {"code": "print(1)"}
        )
        assert result.decision == "ALLOWED"

    def test_code_execute_denied_when_not_granted(self) -> None:
        bundle = make_policy_bundle(capabilities=[{"name": "File.Read"}, {"name": "LLM.Call"}])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "ExecuteCode", CapabilityName.CODE_EXECUTE, {"code": "print(1)"}
        )
        assert result.decision == "DENIED"
        assert "not granted" in result.reason.lower()

    def test_code_execute_denied_when_python_not_allowed(self) -> None:
        bundle = make_policy_bundle(
            capabilities=[
                {
                    "name": "Code.Execute",
                    "allowedLanguages": ["javascript"],
                },
                {"name": "LLM.Call"},
            ]
        )
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "ExecuteCode", CapabilityName.CODE_EXECUTE, {"code": "print(1)"}
        )
        assert result.decision == "DENIED"
        assert "python" in result.reason.lower()

    def test_code_execute_allowed_when_python_in_list(self) -> None:
        bundle = make_policy_bundle(
            capabilities=[
                {
                    "name": "Code.Execute",
                    "allowedLanguages": ["python", "javascript"],
                },
                {"name": "LLM.Call"},
            ]
        )
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "ExecuteCode", CapabilityName.CODE_EXECUTE, {"code": "print(1)"}
        )
        assert result.decision == "ALLOWED"

    def test_code_execute_allowed_when_no_language_restriction(self) -> None:
        """When allowedLanguages is None, defaults to ['python'] — which allows Python."""
        bundle = make_policy_bundle(capabilities=[{"name": "Code.Execute"}, {"name": "LLM.Call"}])
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "ExecuteCode", CapabilityName.CODE_EXECUTE, {"code": "print(1)"}
        )
        assert result.decision == "ALLOWED"

    def test_code_execute_requires_approval(self) -> None:
        bundle = make_policy_bundle(
            capabilities=[
                {
                    "name": "Code.Execute",
                    "requiresApproval": True,
                    "approvalRuleId": "code-rule",
                },
                {"name": "LLM.Call"},
            ],
            approval_rules=[
                {
                    "approvalRuleId": "code-rule",
                    "title": "Code execution",
                    "description": "Approve code",
                }
            ],
        )
        enforcer = PolicyEnforcer(bundle)
        result = enforcer.check_tool_call(
            "ExecuteCode", CapabilityName.CODE_EXECUTE, {"code": "print(1)"}
        )
        assert result.decision == "APPROVAL_REQUIRED"
