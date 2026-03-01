"""Creates and configures an ADK LlmAgent for Cowork."""

from __future__ import annotations

from typing import TYPE_CHECKING

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from agent_host.agent.callbacks import (
    make_after_tool_callback,
    make_before_model_callback,
    make_before_tool_callback,
)
from agent_host.agent.tool_adapter import adapt_tools
from agent_host.budget.token_budget import TokenBudget
from agent_host.policy.policy_enforcer import PolicyEnforcer

if TYPE_CHECKING:
    from cowork_platform.policy_bundle import PolicyBundle

    from agent_host.config import AgentHostConfig
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.session.workspace_client import WorkspaceClient
    from tool_runtime import ToolRouter
    from tool_runtime.models import ExecutionContext

SYSTEM_PROMPT = """You are Cowork, a capable AI assistant running on the user's desktop.
You have access to tools for reading files, writing files, deleting files,
running shell commands, and making HTTP requests.

Guidelines:
- Use tools to accomplish the user's requests. Always verify your work.
- When writing files, show the user what you plan to write before doing so.
- When running commands, explain what the command does.
- If a tool call is denied by policy, explain why and suggest alternatives.
- Be concise and helpful. Focus on completing the task efficiently.
- If you encounter an error, try to recover or suggest a fix.
"""


def create_agent(
    config: AgentHostConfig,
    policy_bundle: PolicyBundle,
    tool_router: ToolRouter,
    event_emitter: EventEmitter | None = None,
    workspace_client: WorkspaceClient | None = None,
    execution_context: ExecutionContext | None = None,
    session_id: str = "",
    task_id: str = "",
) -> tuple[LlmAgent, TokenBudget]:
    """Create an LlmAgent configured for Cowork.

    Args:
        config: Agent host configuration.
        policy_bundle: Session policy bundle for enforcement.
        tool_router: Tool router for executing tools.
        event_emitter: Optional event emitter for notifications.
        workspace_client: Optional workspace client for artifact uploads.
        execution_context: Optional execution context for tool constraints.
        session_id: Current session ID.
        task_id: Current task ID.

    Returns:
        Tuple of (LlmAgent, TokenBudget) — the budget is needed for
        recording actual usage after LLM responses.
    """
    # Initialize policy enforcer and token budget
    policy_enforcer = PolicyEnforcer(policy_bundle)
    token_budget = TokenBudget(
        max_session_tokens=policy_bundle.llmPolicy.maxSessionTokens,
    )

    # Configure LLM model via LiteLLM
    model = LiteLlm(
        model=config.llm_model,
        api_base=config.llm_gateway_endpoint,
        api_key=config.llm_gateway_auth_token,
    )

    # Adapt ToolRouter tools to ADK FunctionTools
    adk_tools = adapt_tools(
        tool_router=tool_router,
        policy_enforcer=policy_enforcer,
        execution_context=execution_context,
        session_id=session_id,
        task_id=task_id,
    )

    # Create callbacks
    before_tool = make_before_tool_callback(
        policy_enforcer=policy_enforcer,
        event_emitter=event_emitter,
    )
    before_model = make_before_model_callback(
        policy_enforcer=policy_enforcer,
        token_budget=token_budget,
    )
    after_tool = make_after_tool_callback(
        workspace_client=workspace_client,
        event_emitter=event_emitter,
    )

    # Create the agent
    agent = LlmAgent(
        name="cowork_agent",
        model=model,
        instruction=SYSTEM_PROMPT,
        tools=adk_tools,
        before_tool_callback=before_tool,
        before_model_callback=before_model,
        after_tool_callback=after_tool,
    )

    return agent, token_budget
