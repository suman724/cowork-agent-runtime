"""Creates and configures an ADK LlmAgent for Cowork."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from agent_host.agent.artifact_store import PendingArtifactStore
from agent_host.agent.callbacks import (
    make_after_tool_callback,
    make_before_model_callback,
    make_before_tool_callback,
)
from agent_host.agent.file_change_tracker import FileChangeTracker
from agent_host.agent.tool_adapter import ToolExecutionDeps, adapt_tools
from agent_host.approval.approval_gate import ApprovalGate
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


@dataclass
class AgentComponents:
    """All components created by ``create_agent``.

    Returned to ``SessionManager`` so it can wire up JSON-RPC handlers
    (approval delivery, patch preview, etc.).
    """

    agent: LlmAgent
    token_budget: TokenBudget
    approval_gate: ApprovalGate
    artifact_store: PendingArtifactStore
    file_change_tracker: FileChangeTracker


def create_agent(
    config: AgentHostConfig,
    policy_bundle: PolicyBundle,
    tool_router: ToolRouter,
    event_emitter: EventEmitter | None = None,
    workspace_client: WorkspaceClient | None = None,
    execution_context: ExecutionContext | None = None,
    session_id: str = "",
    task_id: str = "",
    workspace_id: str = "",
) -> AgentComponents:
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
        workspace_id: Current workspace ID.

    Returns:
        AgentComponents containing the agent and all helper objects
        needed by SessionManager.
    """
    # Initialize policy enforcer and token budget
    policy_enforcer = PolicyEnforcer(policy_bundle)
    max_tokens = policy_bundle.llmPolicy.maxSessionTokens if policy_bundle.llmPolicy else 100_000
    token_budget = TokenBudget(max_session_tokens=max_tokens)

    # Create helper objects
    approval_gate = ApprovalGate()
    artifact_store = PendingArtifactStore()
    file_change_tracker = FileChangeTracker()

    # Bundle dependencies for tool functions
    deps = ToolExecutionDeps(
        tool_router=tool_router,
        policy_enforcer=policy_enforcer,
        event_emitter=event_emitter,
        approval_gate=approval_gate,
        artifact_store=artifact_store,
        file_change_tracker=file_change_tracker,
        approval_timeout=float(config.approval_timeout_seconds),
        session_id=session_id,
        task_id=task_id,
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
        deps=deps,
    )

    # Create callbacks
    before_tool = make_before_tool_callback(
        policy_enforcer=policy_enforcer,
        event_emitter=event_emitter,
    )
    before_model = make_before_model_callback(
        policy_enforcer=policy_enforcer,
        token_budget=token_budget,
        event_emitter=event_emitter,
        max_context_tokens=config.max_context_tokens,
    )
    after_tool = make_after_tool_callback(
        workspace_client=workspace_client,
        event_emitter=event_emitter,
        artifact_store=artifact_store,
        session_id=session_id,
        workspace_id=workspace_id,
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

    return AgentComponents(
        agent=agent,
        token_budget=token_budget,
        approval_gate=approval_gate,
        artifact_store=artifact_store,
        file_change_tracker=file_change_tracker,
    )
