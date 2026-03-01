"""Adapts ToolRouter tools to ADK FunctionTool / LongRunningFunctionTool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cowork_platform.tool_request import ToolRequest
from google.adk.tools import FunctionTool, LongRunningFunctionTool  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from agent_host.policy.policy_enforcer import PolicyEnforcer
    from tool_runtime import ToolRouter
    from tool_runtime.models import ExecutionContext

# Tool name → capability name mapping
TOOL_CAPABILITY_MAP: dict[str, str] = {
    "ReadFile": "File.Read",
    "WriteFile": "File.Write",
    "DeleteFile": "File.Delete",
    "RunCommand": "Shell.Exec",
    "HttpRequest": "Network.Http",
}


def get_capability_for_tool(tool_name: str) -> str:
    """Return the capability name for a given tool, or empty string if unknown."""
    return TOOL_CAPABILITY_MAP.get(tool_name, "")


def adapt_tools(
    tool_router: ToolRouter,
    policy_enforcer: PolicyEnforcer,
    execution_context: ExecutionContext | None = None,
    session_id: str = "",
    task_id: str = "",
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
            tool_router=tool_router,
            tool_name=tool_name,
            capability_name=capability_name,
            description=tool_def.description,
            execution_context=execution_context,
            session_id=session_id,
            task_id=task_id,
        )

        if requires_approval:
            tools.append(LongRunningFunctionTool(func=exec_fn))
        else:
            tools.append(FunctionTool(func=exec_fn))

    return tools


def _make_tool_function(
    tool_router: ToolRouter,
    tool_name: str,
    capability_name: str,
    description: str,
    execution_context: ExecutionContext | None,
    session_id: str,
    task_id: str,
) -> Any:
    """Create an async function that dispatches to ToolRouter.

    The function signature is dynamically set so ADK can inspect it.
    We use **kwargs to accept any arguments the LLM passes.
    """

    async def tool_fn(**kwargs: Any) -> dict[str, Any]:
        request = ToolRequest(
            toolName=tool_name,
            arguments=kwargs,
            sessionId=session_id,
            taskId=task_id,
            stepId="",
            capability=capability_name or None,
        )
        try:
            result = await tool_router.execute(request, execution_context)
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
