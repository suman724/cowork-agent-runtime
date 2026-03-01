"""Adapts ToolRouter tools to ADK FunctionTool / LongRunningFunctionTool."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from cowork_platform.tool_request import ToolRequest
from google.adk.tools import FunctionTool, LongRunningFunctionTool  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from agent_host.agent.artifact_store import PendingArtifactStore
    from agent_host.agent.file_change_tracker import FileChangeTracker
    from agent_host.approval.approval_gate import ApprovalGate
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.policy.policy_enforcer import PolicyEnforcer
    from tool_runtime import ToolRouter
    from tool_runtime.models import ExecutionContext

logger = structlog.get_logger()

# Tool name → capability name mapping
TOOL_CAPABILITY_MAP: dict[str, str] = {
    "ReadFile": "File.Read",
    "WriteFile": "File.Write",
    "DeleteFile": "File.Delete",
    "RunCommand": "Shell.Exec",
    "HttpRequest": "Network.Http",
}

# Tools that mutate files — used for file change tracking
_FILE_WRITE_TOOLS = {"WriteFile"}
_FILE_DELETE_TOOLS = {"DeleteFile"}


@dataclass
class ToolExecutionDeps:
    """Bundles dependencies for tool execution functions."""

    tool_router: ToolRouter
    policy_enforcer: PolicyEnforcer
    event_emitter: EventEmitter | None = None
    approval_gate: ApprovalGate | None = None
    artifact_store: PendingArtifactStore | None = None
    file_change_tracker: FileChangeTracker | None = None
    approval_timeout: float = 300.0
    session_id: str = ""
    task_id: str = ""
    # requires_approval per tool: tool_name -> bool
    _approval_map: dict[str, bool] = field(default_factory=dict)


def get_capability_for_tool(tool_name: str) -> str:
    """Return the capability name for a given tool, or empty string if unknown."""
    return TOOL_CAPABILITY_MAP.get(tool_name, "")


def adapt_tools(
    tool_router: ToolRouter,
    policy_enforcer: PolicyEnforcer,
    execution_context: ExecutionContext | None = None,
    session_id: str = "",
    task_id: str = "",
    deps: ToolExecutionDeps | None = None,
) -> list[FunctionTool | LongRunningFunctionTool]:
    """Convert each ToolRouter tool into an ADK FunctionTool.

    For each ToolDefinition from tool_router.get_available_tools():
    - Create an async function that calls tool_router.execute()
    - Wrap it as FunctionTool with the tool's input schema
    - If the tool's capability has requiresApproval=True in the policy,
      wrap as LongRunningFunctionTool instead
    """
    tools: list[FunctionTool | LongRunningFunctionTool] = []

    for tool_def in tool_router.get_available_tools():
        tool_name = tool_def.toolName
        capability_name = get_capability_for_tool(tool_name)

        # Check if this tool requires approval
        requires_approval = False
        if capability_name:
            cap = policy_enforcer.get_capability(capability_name)
            if cap and cap.requiresApproval:
                requires_approval = True

        # Create the async execution function
        exec_fn = _make_tool_function(
            tool_name=tool_name,
            capability_name=capability_name,
            description=tool_def.description,
            execution_context=execution_context,
            requires_approval=requires_approval,
            deps=deps,
            # Fallback params for backward compatibility (no deps)
            tool_router=tool_router,
            session_id=session_id,
            task_id=task_id,
        )

        if requires_approval:
            tools.append(LongRunningFunctionTool(func=exec_fn))
        else:
            tools.append(FunctionTool(func=exec_fn))

    return tools


def _make_tool_function(
    tool_name: str,
    capability_name: str,
    description: str,
    execution_context: ExecutionContext | None,
    requires_approval: bool,
    deps: ToolExecutionDeps | None,
    tool_router: ToolRouter,
    session_id: str,
    task_id: str,
) -> Any:
    """Create an async function that dispatches to ToolRouter.

    The function signature is dynamically set so ADK can inspect it.
    We use **kwargs to accept any arguments the LLM passes.
    """
    # Resolve effective values from deps or fallback params
    _router = deps.tool_router if deps else tool_router
    _session_id = deps.session_id if deps else session_id
    _task_id = deps.task_id if deps else task_id

    async def tool_fn(**kwargs: Any) -> dict[str, Any]:
        # --- Approval gate ---
        if requires_approval and deps and deps.approval_gate and deps.event_emitter:
            approval_id = str(uuid.uuid4())
            deps.event_emitter.emit_approval_requested(
                approval_id=approval_id,
                risk_level="medium",
                tool_name=tool_name,
                action_summary=f"{tool_name} ({capability_name})",
                session_id=_session_id,
                task_id=_task_id,
                title=f"Approve {tool_name}",
            )
            decision = await deps.approval_gate.request_approval(
                approval_id, timeout=deps.approval_timeout
            )
            if decision != "approved":
                logger.info(
                    "tool_call_denied_by_approval",
                    tool_name=tool_name,
                    decision=decision,
                    approval_id=approval_id,
                )
                return {
                    "status": "denied",
                    "output": "",
                    "error": {
                        "code": "APPROVAL_DENIED",
                        "message": f"User {decision} the {tool_name} tool call",
                    },
                }

        # --- File change tracking: capture old content BEFORE execution ---
        old_content: str | None = None
        file_path_arg: str | None = None
        if deps and deps.file_change_tracker:
            if tool_name in _FILE_WRITE_TOOLS:
                file_path_arg = kwargs.get("path") or kwargs.get("filePath")
                if file_path_arg:
                    try:
                        old_content = Path(file_path_arg).read_text(encoding="utf-8")
                    except (FileNotFoundError, OSError):
                        old_content = None
            elif tool_name in _FILE_DELETE_TOOLS:
                file_path_arg = kwargs.get("path") or kwargs.get("filePath")
                if file_path_arg:
                    try:
                        old_content = Path(file_path_arg).read_text(encoding="utf-8")
                    except (FileNotFoundError, OSError):
                        old_content = ""

        # --- Execute the tool ---
        request = ToolRequest(
            toolName=tool_name,
            arguments=kwargs,
            sessionId=_session_id,
            taskId=_task_id,
            stepId="",
            capability=capability_name or None,
        )
        try:
            result = await _router.execute(request, execution_context)

            # --- File change tracking: record changes AFTER execution ---
            if deps and deps.file_change_tracker and file_path_arg:
                if tool_name in _FILE_WRITE_TOOLS:
                    try:
                        new_content = Path(file_path_arg).read_text(encoding="utf-8")
                    except (FileNotFoundError, OSError):
                        new_content = kwargs.get("content", "")
                    deps.file_change_tracker.record_write(
                        _task_id, file_path_arg, old_content, new_content
                    )
                elif tool_name in _FILE_DELETE_TOOLS:
                    deps.file_change_tracker.record_delete(
                        _task_id, file_path_arg, old_content or ""
                    )

            # --- Artifact store: stash artifacts for after_tool_callback ---
            if deps and deps.artifact_store and result.artifacts:
                deps.artifact_store.store(tool_name, result.artifacts)

            return {
                "status": result.tool_result.status,
                "output": result.tool_result.outputText or "",
                "error": (
                    result.tool_result.error.model_dump() if result.tool_result.error else None
                ),
            }
        except Exception as exc:
            return {
                "status": "failed",
                "output": "",
                "error": {"code": "TOOL_EXECUTION_FAILED", "message": str(exc)},
            }

    # Set function metadata for ADK
    tool_fn.__name__ = tool_name
    tool_fn.__qualname__ = tool_name
    tool_fn.__doc__ = description

    return tool_fn
