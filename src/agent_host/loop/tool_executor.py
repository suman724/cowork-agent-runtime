"""ToolExecutor — policy-checked tool dispatch with approval, tracking, and artifacts."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from agent_host.loop.models import ToolCallResult

if TYPE_CHECKING:
    from agent_host.agent.file_change_tracker import FileChangeTracker
    from agent_host.approval.approval_client import ApprovalClient
    from agent_host.approval.approval_gate import ApprovalGate
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.llm.models import ToolCallMessage
    from agent_host.policy.policy_enforcer import PolicyEnforcer
    from agent_host.session.workspace_client import WorkspaceClient
    from tool_runtime import ToolRouter
    from tool_runtime.models import ExecutionContext

logger = structlog.get_logger()

# Tool name → capability name mapping (carried from tool_adapter.py)
TOOL_CAPABILITY_MAP: dict[str, str] = {
    "ReadFile": "File.Read",
    "WriteFile": "File.Write",
    "DeleteFile": "File.Delete",
    "EditFile": "File.Write",
    "MultiEdit": "File.Write",
    "CreateDirectory": "File.Write",
    "MoveFile": "File.Write",
    "ListDirectory": "File.Read",
    "FindFiles": "File.Read",
    "GrepFiles": "File.Read",
    "ViewImage": "File.Read",
    "RunCommand": "Shell.Exec",
    "HttpRequest": "Network.Http",
    "FetchUrl": "Network.Http",
    "WebSearch": "Search.Web",
    "ExecuteCode": "Code.Execute",
}

# Tools that mutate files
_FILE_WRITE_TOOLS = {"WriteFile", "EditFile", "MultiEdit"}
_FILE_DELETE_TOOLS = {"DeleteFile"}
_FILE_MOVE_TOOLS = {"MoveFile"}

# Maximum length for tool output stored in history messages
_MAX_TOOL_OUTPUT_LENGTH = 4000

# Tools safe to run in parallel (read-only, no side effects)
_PARALLELIZABLE_TOOLS = {
    "ReadFile",
    "ListDirectory",
    "FindFiles",
    "GrepFiles",
    "ViewImage",
    "FetchUrl",
    "WebSearch",
}

# File.Write tools that are parallelizable if targeting different paths
_PARALLEL_IF_DIFFERENT_PATH = {"WriteFile", "EditFile", "MultiEdit"}

# Tools allowed in plan mode (read-only exploration)
PLAN_MODE_ALLOWED_TOOLS = {
    "ReadFile",
    "ListDirectory",
    "FindFiles",
    "GrepFiles",
    "ViewImage",
    "FetchUrl",
    "WebSearch",
}


class ToolExecutor:
    """Executes tool calls with policy enforcement, approval, file tracking, and artifacts.

    Consolidates logic from the prior tool_adapter.py, callbacks.py, and artifact_store.py.
    """

    def __init__(
        self,
        tool_router: ToolRouter,
        policy_enforcer: PolicyEnforcer,
        event_emitter: EventEmitter | None = None,
        approval_gate: ApprovalGate | None = None,
        approval_client: ApprovalClient | None = None,
        file_change_tracker: FileChangeTracker | None = None,
        workspace_client: WorkspaceClient | None = None,
        execution_context: ExecutionContext | None = None,
        session_id: str = "",
        workspace_id: str = "",
        tenant_id: str = "",
        user_id: str = "",
        approval_timeout: float = 300.0,
        plan_mode: bool = False,
        plan_mode_locked: bool = False,
    ) -> None:
        self._tool_router = tool_router
        self._policy_enforcer = policy_enforcer
        self._event_emitter = event_emitter
        self._approval_gate = approval_gate
        self._approval_client = approval_client
        self._file_change_tracker = file_change_tracker
        self._workspace_client = workspace_client
        self._execution_context = execution_context
        self._session_id = session_id
        self._workspace_id = workspace_id
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._approval_timeout = approval_timeout
        self._plan_mode = plan_mode
        self._plan_mode_locked = plan_mode_locked

    @property
    def plan_mode(self) -> bool:
        """Whether plan mode is currently active."""
        return self._plan_mode

    @plan_mode.setter
    def plan_mode(self, value: bool) -> None:
        if not self._plan_mode_locked:
            self._plan_mode = value

    @property
    def plan_mode_locked(self) -> bool:
        """Whether plan mode is hard-locked (planOnly=true)."""
        return self._plan_mode_locked

    async def execute_tool_calls(
        self,
        calls: list[ToolCallMessage],
        task_id: str,
        step_id: str = "",
    ) -> list[ToolCallResult]:
        """Execute tool calls with parallel grouping.

        Safe-to-parallelize tools (reads, fetches) run concurrently within groups.
        Non-parallelizable tools (writes, shell, deletes) serialize as barriers.

        Per call:
        1. Policy check (reuses PolicyEnforcer.check_tool_call)
        2. Emit tool_requested event
        3. Approval gate if required (reuses ApprovalGate)
        4. File change tracking before/after (reuses FileChangeTracker)
        5. Execute via ToolRouter
        6. Artifact upload (fire-and-forget, reuses WorkspaceClient)
        7. Emit tool_completed event
        """
        groups = self._partition_parallel_groups(calls)
        results: list[ToolCallResult] = []

        for group in groups:
            if len(group) == 1:
                results.append(await self._execute_single(group[0], task_id, step_id=step_id))
            else:
                group_results = await asyncio.gather(
                    *(self._execute_single(c, task_id, step_id=step_id) for c in group)
                )
                results.extend(group_results)

        return results

    def _partition_parallel_groups(
        self, calls: list[ToolCallMessage]
    ) -> list[list[ToolCallMessage]]:
        """Partition tool calls into ordered groups for parallel/serial execution."""
        groups: list[list[ToolCallMessage]] = []
        current_batch: list[ToolCallMessage] = []
        current_batch_paths: set[str] = set()

        for call in calls:
            if call.name in _PARALLELIZABLE_TOOLS:
                current_batch.append(call)
            elif call.name in _PARALLEL_IF_DIFFERENT_PATH:
                path = call.arguments.get("path", "")
                if path and path not in current_batch_paths:
                    current_batch.append(call)
                    current_batch_paths.add(path)
                else:
                    if current_batch:
                        groups.append(current_batch)
                        current_batch = []
                        current_batch_paths = set()
                    groups.append([call])
            else:
                if current_batch:
                    groups.append(current_batch)
                    current_batch = []
                    current_batch_paths = set()
                groups.append([call])

        if current_batch:
            groups.append(current_batch)

        return groups

    async def _execute_single(
        self,
        call: ToolCallMessage,
        task_id: str,
        step_id: str = "",
    ) -> ToolCallResult:
        """Execute a single tool call with full lifecycle."""
        tool_name = call.name
        arguments = call.arguments
        capability_name = TOOL_CAPABILITY_MAP.get(tool_name, "")

        # Plan mode enforcement — block write/exec tools
        if self._plan_mode and tool_name not in PLAN_MODE_ALLOWED_TOOLS:
            logger.info("tool_blocked_plan_mode", tool_name=tool_name)
            if self._event_emitter:
                self._event_emitter.emit_tool_requested(
                    tool_name, capability_name, arguments, tool_call_id=call.id
                )
                self._event_emitter.emit_tool_completed(
                    tool_name,
                    "denied",
                    tool_call_id=call.id,
                    error="Blocked in plan mode",
                )
            denial_msg = json.dumps(
                {
                    "status": "denied",
                    "error": {
                        "code": "PLAN_MODE_RESTRICTED",
                        "message": (
                            f"{tool_name} is not available in plan mode. "
                            "Call ExitPlanMode first to enable write operations."
                        ),
                    },
                }
            )
            return ToolCallResult(
                tool_call_id=call.id,
                tool_name=tool_name,
                status="denied",
                result_text=denial_msg,
                arguments=arguments,
            )

        # 1. Policy check
        if capability_name:
            # Emit tool_requested BEFORE policy check so the desktop always creates
            # a ToolCallCard — even for denied tools (card shows "Denied" badge).
            if self._event_emitter:
                self._event_emitter.emit_tool_requested(
                    tool_name, capability_name, arguments, tool_call_id=call.id
                )

            check = self._policy_enforcer.check_tool_call(tool_name, capability_name, arguments)
            if check.decision == "DENIED":
                logger.warning(
                    "tool_call_denied",
                    tool_name=tool_name,
                    capability=capability_name,
                    reason=check.reason,
                )
                if self._event_emitter:
                    self._event_emitter.emit_tool_completed(
                        tool_name,
                        "denied",
                        tool_call_id=call.id,
                        error=check.reason,
                    )
                denial_msg = json.dumps(
                    {
                        "status": "denied",
                        "error": {"code": "CAPABILITY_DENIED", "message": check.reason},
                    }
                )
                return ToolCallResult(
                    tool_call_id=call.id,
                    tool_name=tool_name,
                    status="denied",
                    result_text=denial_msg,
                    arguments=arguments,
                )

            # 2. Approval gate if required
            if check.decision == "APPROVAL_REQUIRED":
                result = await self._handle_approval(call, task_id, capability_name)
                if result is not None:
                    return result
        else:
            # Unknown tool — still emit requested event
            if self._event_emitter:
                self._event_emitter.emit_tool_requested(
                    tool_name, "", arguments, tool_call_id=call.id
                )

        # 4. File change tracking: capture old content BEFORE execution
        old_content, file_path_arg = self._capture_pre_state(tool_name, arguments)

        # 5. Execute via ToolRouter
        try:
            from cowork_platform.tool_request import ToolRequest

            request = ToolRequest(
                toolName=tool_name,
                arguments=arguments,
                sessionId=self._session_id,
                taskId=task_id,
                stepId=step_id,
                capability=capability_name or None,
            )
            exec_result = await self._tool_router.execute(request, self._execution_context)

            # Record file changes AFTER execution
            self._record_file_changes(tool_name, task_id, file_path_arg, old_content, arguments)

            # 6. Artifact upload (fire-and-forget)
            if exec_result.artifacts:
                self._upload_artifacts(exec_result.artifacts, tool_name)

            # 7. Emit tool_completed event
            status = exec_result.tool_result.status
            output = exec_result.tool_result.outputText or ""
            error = exec_result.tool_result.error
            if self._event_emitter:
                self._event_emitter.emit_tool_completed(
                    tool_name,
                    status,
                    tool_call_id=call.id,
                    result=output[:_MAX_TOOL_OUTPUT_LENGTH] if output else None,
                    error=error.message if error else None,
                )
            result_dict: dict[str, Any] = {"status": status, "output": output}
            if error:
                result_dict["error"] = error.model_dump()
            result_text = json.dumps(result_dict, default=str)

            # Build data URI for multimodal image content
            image_url: str | None = None
            if exec_result.image_content is not None:
                ic = exec_result.image_content
                image_url = f"data:{ic.media_type};base64,{ic.base64_data}"

            return ToolCallResult(
                tool_call_id=call.id,
                tool_name=tool_name,
                status=status,
                result_text=result_text,
                arguments=arguments,
                artifacts=exec_result.artifacts or None,
                image_url=image_url,
            )

        except Exception as exc:
            logger.warning(
                "tool_execution_failed",
                tool_name=tool_name,
                error=str(exc),
                exc_info=True,
            )
            if self._event_emitter:
                self._event_emitter.emit_tool_completed(
                    tool_name,
                    "failed",
                    tool_call_id=call.id,
                    error=str(exc),
                )
            error_text = json.dumps(
                {
                    "status": "failed",
                    "error": {"code": "TOOL_EXECUTION_FAILED", "message": str(exc)},
                }
            )
            return ToolCallResult(
                tool_call_id=call.id,
                tool_name=tool_name,
                status="failed",
                result_text=error_text,
                arguments=arguments,
            )

    async def _handle_approval(
        self,
        call: ToolCallMessage,
        task_id: str,
        capability_name: str,
    ) -> ToolCallResult | None:
        """Handle the approval flow. Returns ToolCallResult if denied, None if approved."""
        if not self._approval_gate or not self._event_emitter:
            return None

        approval_id = str(uuid.uuid4())
        action_summary = f"{call.name} ({capability_name})"
        risk_level = "medium"

        self._event_emitter.emit_approval_requested(
            approval_id=approval_id,
            risk_level=risk_level,
            tool_name=call.name,
            action_summary=action_summary,
            session_id=self._session_id,
            task_id=task_id,
            title=f"Approve {call.name}",
        )

        decision = await self._approval_gate.request_approval(
            approval_id, timeout=self._approval_timeout
        )

        # Persist decision to Approval Service (fire-and-forget)
        self._persist_approval_decision(
            approval_id=approval_id,
            session_id=self._session_id,
            task_id=task_id,
            decision=decision,
            action_summary=action_summary,
            risk_level=risk_level,
        )

        if decision != "approved":
            logger.info(
                "tool_call_denied_by_approval",
                tool_name=call.name,
                decision=decision,
                approval_id=approval_id,
            )
            if self._event_emitter:
                self._event_emitter.emit_tool_completed(
                    call.name,
                    "denied",
                    tool_call_id=call.id,
                    error=f"User {decision} the {call.name} tool call",
                )
            denial_text = json.dumps(
                {
                    "status": "denied",
                    "error": {
                        "code": "APPROVAL_DENIED",
                        "message": f"User {decision} the {call.name} tool call",
                    },
                }
            )
            return ToolCallResult(
                tool_call_id=call.id,
                tool_name=call.name,
                status="denied",
                result_text=denial_text,
                arguments=call.arguments,
            )

        return None  # Approved — continue with execution

    def _persist_approval_decision(
        self,
        *,
        approval_id: str,
        session_id: str,
        task_id: str,
        decision: str,
        action_summary: str,
        risk_level: str,
    ) -> None:
        """Fire-and-forget: persist the approval decision to the Approval Service."""
        if not self._approval_client:
            return

        from datetime import UTC, datetime

        self._approval_client.persist_decision_background(
            approval_id=approval_id,
            session_id=session_id,
            task_id=task_id,
            user_id=self._user_id,
            tenant_id=self._tenant_id,
            workspace_id=self._workspace_id,
            decision=decision,
            action_summary=action_summary,
            risk_level=risk_level,
            client_timestamp=datetime.now(UTC),
        )

    def _capture_pre_state(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        """Capture file content before mutation for change tracking."""
        if not self._file_change_tracker:
            return None, None

        file_path_arg: str | None = None
        old_content: str | None = None

        if tool_name in _FILE_WRITE_TOOLS:
            file_path_arg = arguments.get("path") or arguments.get("filePath")
            if file_path_arg:
                try:
                    old_content = Path(file_path_arg).read_text(encoding="utf-8")
                except (FileNotFoundError, OSError):
                    old_content = None
        elif tool_name in _FILE_DELETE_TOOLS:
            file_path_arg = arguments.get("path") or arguments.get("filePath")
            if file_path_arg:
                try:
                    old_content = Path(file_path_arg).read_text(encoding="utf-8")
                except (FileNotFoundError, OSError):
                    old_content = ""
        elif tool_name in _FILE_MOVE_TOOLS:
            file_path_arg = arguments.get("source", "")
            if file_path_arg:
                try:
                    old_content = Path(file_path_arg).read_text(encoding="utf-8")
                except (FileNotFoundError, OSError):
                    old_content = None

        return old_content, file_path_arg

    def _record_file_changes(
        self,
        tool_name: str,
        task_id: str,
        file_path_arg: str | None,
        old_content: str | None,
        arguments: dict[str, Any],
    ) -> None:
        """Record file changes after successful execution."""
        if not self._file_change_tracker or not file_path_arg:
            return

        if tool_name in _FILE_WRITE_TOOLS:
            try:
                new_content = Path(file_path_arg).read_text(encoding="utf-8")
            except (FileNotFoundError, OSError):
                new_content = arguments.get("content", "")
            self._file_change_tracker.record_write(task_id, file_path_arg, old_content, new_content)
        elif tool_name in _FILE_DELETE_TOOLS:
            self._file_change_tracker.record_delete(task_id, file_path_arg, old_content or "")
        elif tool_name in _FILE_MOVE_TOOLS:
            source = arguments.get("source", "")
            dest = arguments.get("destination", "")
            # Record deletion at source
            self._file_change_tracker.record_delete(task_id, source, old_content or "")
            # Record creation at destination
            try:
                new_content = Path(dest).read_text(encoding="utf-8")
            except (FileNotFoundError, OSError):
                new_content = old_content or ""
            self._file_change_tracker.record_write(task_id, dest, None, new_content)

    def _upload_artifacts(
        self,
        artifacts: list[Any],
        tool_name: str,
    ) -> None:
        """Upload artifacts fire-and-forget."""
        import asyncio

        if not self._workspace_client or not self._workspace_id:
            return

        for artifact in artifacts:
            try:
                asyncio.create_task(  # noqa: RUF006
                    self._workspace_client.upload_artifact(
                        workspace_id=self._workspace_id,
                        session_id=self._session_id,
                        artifact_data=artifact.data,
                        artifact_type=artifact.artifact_type,
                        artifact_name=artifact.artifact_name,
                        content_type=artifact.media_type,
                    )
                )
            except Exception:
                logger.warning(
                    "artifact_upload_failed",
                    tool_name=tool_name,
                    artifact_name=artifact.artifact_name,
                    exc_info=True,
                )

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool definitions, filtered for plan mode if active."""
        tool_defs: list[dict[str, Any]] = []
        for tool_def in self._tool_router.get_available_tools():
            # Filter out write/exec tools in plan mode
            if self._plan_mode and tool_def.toolName not in PLAN_MODE_ALLOWED_TOOLS:
                continue

            fn_def: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool_def.toolName,
                    "description": tool_def.description,
                },
            }
            if tool_def.inputSchema:
                fn_def["function"]["parameters"] = tool_def.inputSchema
            else:
                fn_def["function"]["parameters"] = {"type": "object", "properties": {}}

            tool_defs.append(fn_def)
        return tool_defs
