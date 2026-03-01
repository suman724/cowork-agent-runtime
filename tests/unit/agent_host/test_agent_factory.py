"""Tests for agent_factory — creating ADK agents with correct configuration."""

from __future__ import annotations

from unittest.mock import MagicMock

from cowork_platform.tool_definition import ToolDefinition
from google.adk.agents import LlmAgent

from agent_host.agent.agent_factory import SYSTEM_PROMPT, create_agent
from agent_host.budget.token_budget import TokenBudget
from agent_host.config import AgentHostConfig
from tests.fixtures.policy_bundles import make_policy_bundle


def _make_config() -> AgentHostConfig:
    return AgentHostConfig(
        llm_gateway_endpoint="https://llm.example.com",
        llm_gateway_auth_token="test-token",
        session_service_url="https://sessions.example.com",
        workspace_service_url="https://workspace.example.com",
        checkpoint_dir="/tmp/checkpoints",
    )


def _make_tool_router() -> MagicMock:
    router = MagicMock()
    router.get_available_tools.return_value = [
        ToolDefinition(
            toolName="ReadFile",
            description="Read a file",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        ToolDefinition(
            toolName="WriteFile",
            description="Write a file",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    ]
    return router


class TestCreateAgent:
    def test_returns_agent_and_budget(self) -> None:
        """create_agent returns an LlmAgent and TokenBudget."""
        config = _make_config()
        bundle = make_policy_bundle()
        router = _make_tool_router()

        agent, budget = create_agent(
            config=config,
            policy_bundle=bundle,
            tool_router=router,
        )

        assert isinstance(agent, LlmAgent)
        assert isinstance(budget, TokenBudget)

    def test_agent_has_correct_name(self) -> None:
        config = _make_config()
        bundle = make_policy_bundle()
        router = _make_tool_router()

        agent, _ = create_agent(config=config, policy_bundle=bundle, tool_router=router)
        assert agent.name == "cowork_agent"

    def test_agent_has_system_prompt(self) -> None:
        config = _make_config()
        bundle = make_policy_bundle()
        router = _make_tool_router()

        agent, _ = create_agent(config=config, policy_bundle=bundle, tool_router=router)
        assert agent.instruction == SYSTEM_PROMPT

    def test_budget_matches_policy(self) -> None:
        config = _make_config()
        bundle = make_policy_bundle(max_session_tokens=50000)
        router = _make_tool_router()

        _, budget = create_agent(config=config, policy_bundle=bundle, tool_router=router)
        assert budget.max_session_tokens == 50000

    def test_agent_has_tools(self) -> None:
        config = _make_config()
        bundle = make_policy_bundle()
        router = _make_tool_router()

        agent, _ = create_agent(config=config, policy_bundle=bundle, tool_router=router)
        assert len(agent.tools) == 2

    def test_agent_has_callbacks(self) -> None:
        config = _make_config()
        bundle = make_policy_bundle()
        router = _make_tool_router()

        agent, _ = create_agent(config=config, policy_bundle=bundle, tool_router=router)
        assert agent.before_tool_callback is not None
        assert agent.before_model_callback is not None
        assert agent.after_tool_callback is not None
