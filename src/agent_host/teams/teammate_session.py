"""TeammateSessionManager — lightweight agent loop for team members."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from agent_host.budget.token_budget import TokenBudget
from agent_host.loop.agent_tools import AgentToolHandler
from agent_host.loop.loop_runtime import LoopRuntime
from agent_host.loop.react_loop import ReactLoop
from agent_host.loop.tool_executor import ToolExecutor
from agent_host.memory.working_memory import WorkingMemory
from agent_host.teams.task_list import SharedTaskList
from agent_host.thread.compactor import DropOldestCompactor
from agent_host.thread.message_thread import MessageThread

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_host.coordination.protocols import (
        ContextInjectionStrategy,
        ToolProviderStrategy,
    )
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.llm.client import LLMClient
    from agent_host.loop.models import LoopResult
    from agent_host.policy.policy_enforcer import PolicyEnforcer
    from agent_host.session.workspace_client import WorkspaceClient
    from tool_runtime import ToolRouter

logger = structlog.get_logger()

_TEAMMATE_SYSTEM_PROMPT = """You are a teammate in a multi-agent team working on a shared project.

Team: {team_name}
Your name: {teammate_name}
Your role: {role}

## Working Directory
You are working in: {workspace_dir}
You share this workspace with other teammates. Stick to the files relevant to your \
assigned tasks. Check the task list to see what others are working on to avoid conflicts.

## Coordination — IMPORTANT
You MUST use the team coordination tools to keep the team in sync:

1. **Start of work**: Call TeamTaskCreate to register what you are working on.
2. **Progress**: Call TeamTaskUpdate with status='in_progress' when you begin a task.
3. **Completion**: Call TeamTaskUpdate with status='completed' and a result summary when done.
4. **Communication**: Use SendTeamMessage to share findings, ask questions, or report to the lead.
5. **Check tasks**: Use TeamTaskList to see what needs to be done and what others are doing.

These tools are how the team stays coordinated and how the user sees your progress. \
Working without updating tasks or sending messages makes you invisible to the team.

## Guidelines
- Focus on your assigned tasks
- When blocked, message the relevant teammate or the lead
- When your current task is done, check the task list for more work
- If the task list is empty and you have no messages, let the lead know you are idle
"""


def _summarize_args(tool_name: str, arguments: dict[str, Any]) -> str:
    """Create a brief human-readable summary of tool arguments."""
    if tool_name in ("ReadFile", "ViewImage"):
        return str(arguments.get("path", ""))
    if tool_name in ("WriteFile", "EditFile", "MultiEdit", "DeleteFile", "MoveFile"):
        return str(arguments.get("path", ""))
    if tool_name == "RunCommand":
        return str(arguments.get("command", ""))[:80]
    if tool_name in ("WebSearch", "FetchUrl"):
        return str(arguments.get("query", "") or arguments.get("url", ""))[:80]
    if tool_name == "GrepFiles":
        return str(arguments.get("pattern", ""))[:60]
    if tool_name == "FindFiles":
        return str(arguments.get("pattern", ""))[:60]
    if tool_name == "ListDirectory":
        return str(arguments.get("path", ""))
    if tool_name in ("TeamTaskCreate", "TeamTaskUpdate", "SendTeamMessage"):
        return str(arguments.get("title", "") or arguments.get("content", ""))[:60]
    if tool_name == "ExecuteCode":
        return str(arguments.get("description", ""))[:60]
    if tool_name == "HttpRequest":
        method = arguments.get("method", "GET")
        url = arguments.get("url", "")
        return f"{method} {url}"[:80]
    # Fallback: first string value
    for v in arguments.values():
        if isinstance(v, str) and v:
            return v[:60]
    return ""


class _TeammateEventProxy:
    """Proxy that routes teammate events to team/* notifications only.

    Teammate events should NOT appear as SessionEvents in the lead's conversation.
    Instead, they are emitted as team/* JSON-RPC notifications:
      - text_chunk → team/teammate_output (streaming text to teammate panel)
      - tool_requested/completed → team/teammate_tool (tool activity indicator)
      - all other session events → suppressed (no-op)

    Only team/* notification methods (notify_raw, emit_team_*) are forwarded
    to the real EventEmitter.
    """

    # Methods that should be forwarded to the delegate (team notifications only)
    _FORWARDED_PREFIXES = ("notify_raw", "emit_team", "emit_teammate")

    def __init__(self, delegate: EventEmitter, team_id: str, teammate_name: str) -> None:
        self._delegate = delegate
        self._team_id = team_id
        self._teammate_name = teammate_name

    def emit_text_chunk(self, task_id: str, text: str, step_id: str | None = None) -> None:  # noqa: ARG002
        """Emit teammate_output only — do NOT forward to lead's conversation."""
        self._delegate.emit_teammate_output(self._team_id, self._teammate_name, text)

    def emit_tool_requested(
        self,
        tool_name: str,
        capability: str,  # noqa: ARG002
        arguments: dict[str, Any],
        tool_call_id: str = "",
        tool_type: str = "tool",  # noqa: ARG002
    ) -> None:
        """Emit teammate_tool only — do NOT forward to lead's conversation."""
        args_summary = _summarize_args(tool_name, arguments)
        self._delegate.emit_teammate_tool(
            self._team_id, self._teammate_name, tool_name, "requested", tool_call_id,
            args=args_summary,
        )

    def emit_tool_completed(
        self,
        tool_name: str,
        status: str,
        tool_call_id: str = "",
        result: str | None = None,  # noqa: ARG002
        error: str | None = None,  # noqa: ARG002
        tool_type: str = "tool",  # noqa: ARG002
    ) -> None:
        """Emit teammate_tool only — do NOT forward to lead's conversation."""
        self._delegate.emit_teammate_tool(
            self._team_id, self._teammate_name, tool_name, status, tool_call_id
        )

    def __getattr__(self, name: str) -> Any:
        """Forward team notification methods; suppress all other session events."""
        if any(name.startswith(prefix) for prefix in self._FORWARDED_PREFIXES):
            return getattr(self._delegate, name)
        # Return a no-op for all session event methods (emit_step_started, etc.)
        return lambda *_args, **_kwargs: None


class TeammateSessionManager:
    """Lightweight session manager for teammate agents.

    Shares: LLMClient, PolicyEnforcer, PolicyBundle, ToolRouter, workspace dir.
    Fresh: MessageThread, TokenBudget, WorkingMemory.
    """

    def __init__(
        self,
        name: str,
        role: str,
        team_name: str,
        llm_client: LLMClient,
        policy_enforcer: PolicyEnforcer,
        tool_router: ToolRouter,
        workspace_dir: str | None,
        tool_provider: ToolProviderStrategy,
        context_injector: ContextInjectionStrategy,
        budget: int = 100_000,
        max_steps: int = 50,
        max_context_tokens: int = 100_000,
        event_emitter: EventEmitter | None = None,
        workspace_client: WorkspaceClient | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
        sync_interval: int = 5,
        team_id: str = "",
        on_activity: Callable[[str], None] | None = None,
        task_list: SharedTaskList | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self._team_name = team_name
        self._team_id = team_id
        self._on_activity = on_activity
        self._task_list = task_list
        self._llm_client = llm_client
        self._policy_enforcer = policy_enforcer
        self._tool_router = tool_router
        self._workspace_dir = workspace_dir
        self._tool_provider = tool_provider
        self._context_injector = context_injector
        self._max_steps = max_steps
        self._max_context_tokens = max_context_tokens
        self._event_emitter = event_emitter
        self._workspace_client = workspace_client
        self._workspace_id = workspace_id
        self._session_id = session_id or f"teammate-{name}"
        self._sync_interval = sync_interval
        self._last_sync_step = 0

        # Fresh per-teammate resources
        self._token_budget = TokenBudget(max_session_tokens=budget)
        self._working_memory = WorkingMemory()
        self._cancel_event = asyncio.Event()

        # Build system prompt
        system_prompt = _TEAMMATE_SYSTEM_PROMPT.format(
            team_name=team_name,
            teammate_name=name,
            role=role,
            workspace_dir=workspace_dir or "(no workspace)",
        )
        self._thread = MessageThread(system_prompt=system_prompt)

        # Task handle for cancellation
        self._task: asyncio.Task[LoopResult | None] | None = None

    @property
    def system_prompt(self) -> str:
        """Return the teammate's system prompt (for testing)."""
        return self._thread._system_prompt

    async def run(self, initial_prompt: str) -> LoopResult | None:
        """Run the teammate agent loop."""
        task_id = f"teammate_{self.name}"

        # Add initial prompt as user message
        self._thread.add_user_message(initial_prompt)

        try:
            from tool_runtime.models import ExecutionContext

            exec_context: ExecutionContext | None = None
            if self._workspace_dir:
                exec_context = ExecutionContext(working_directory=self._workspace_dir)

            tool_executor = ToolExecutor(
                tool_router=self._tool_router,
                policy_enforcer=self._policy_enforcer,
                execution_context=exec_context,
            )

            agent_tool_handler = AgentToolHandler(
                self._working_memory,
                workspace_dir=self._workspace_dir,
                tool_provider=self._tool_provider,
                agent_role="teammate",
                agent_name=self.name,
            )

            compactor = DropOldestCompactor(recency_window=10)

            # Wrap event emitter to forward text chunks as teammate_output
            emitter = self._event_emitter
            if emitter and self._team_id:
                emitter = _TeammateEventProxy(emitter, self._team_id, self.name)  # type: ignore[assignment]

            loop_runtime = LoopRuntime(
                llm_client=self._llm_client,
                tool_executor=tool_executor,
                thread=self._thread,
                compactor=compactor,
                policy_enforcer=self._policy_enforcer,
                token_budget=self._token_budget,
                event_emitter=emitter,
                cancellation_event=self._cancel_event,
                max_context_tokens=self._max_context_tokens,
                working_memory=self._working_memory,
                agent_tool_handler=agent_tool_handler,
                workspace_dir=self._workspace_dir,
                context_injector=self._context_injector,
                on_step_complete=self._on_step_complete,
                agent_name=self.name,
                exit_check=self._check_incomplete_tasks,
            )

            strategy = ReactLoop(loop_runtime, max_steps=self._max_steps)
            result = await strategy.run(task_id)

            # Final history sync on completion
            await self._sync_history(task_id)

            logger.info(
                "teammate_loop_completed",
                teammate_name=self.name,
                team_id=self._team_id,
                reason=result.reason,
                steps=result.step_count,
            )
            return result

        except asyncio.CancelledError:
            logger.info("teammate_cancelled", teammate_name=self.name, team_id=self._team_id)
            return None
        except Exception:
            logger.exception("teammate_loop_error", teammate_name=self.name, team_id=self._team_id)
            return None

    async def _on_step_complete(self, task_id: str, step: int) -> None:
        """Periodic history sync + activity tracking."""
        # Notify coordinator that this teammate is active (resets idle timer)
        if self._on_activity is not None:
            self._on_activity(self.name)

        if (
            self._workspace_client
            and self._workspace_id
            and self._sync_interval > 0
            and step - self._last_sync_step >= self._sync_interval
        ):
            await self._sync_history(task_id)
            self._last_sync_step = step

    async def _sync_history(self, task_id: str) -> None:
        """Upload conversation thread as session history artifact."""
        if not self._workspace_client or not self._workspace_id:
            return
        try:
            from cowork_platform.conversation_message import ConversationMessage

            messages = [ConversationMessage.model_validate(m) for m in self._thread.messages]
            await self._workspace_client.upload_session_history(
                workspace_id=self._workspace_id,
                session_id=self._session_id,
                messages=messages,
                task_id=task_id,
            )
            logger.info(
                "teammate_history_synced",
                teammate_name=self.name,
                team_id=self._team_id,
                task_id=task_id,
                message_count=len(messages),
            )
        except Exception:
            logger.warning(
                "teammate_history_sync_failed",
                teammate_name=self.name,
                team_id=self._team_id,
                exc_info=True,
            )

    async def _check_incomplete_tasks(self) -> str | None:
        """Return a nudge message if this teammate still has incomplete tasks.

        Called by the ReactLoop before allowing natural termination.
        Returns None if exit is allowed (no incomplete work), or a string
        nudge to inject into the conversation to keep the loop alive.
        """
        if self._task_list is None:
            return None

        tasks = await self._task_list.list_tasks()
        my_incomplete = [
            t
            for t in tasks
            if t.status in ("pending", "in_progress", "blocked")
            and (t.assignee == self.name or t.created_by == self.name)
        ]
        if not my_incomplete:
            return None

        task_lines = "\n".join(
            f"- [{t.status}] {t.title} (id={t.task_id})" for t in my_incomplete
        )
        return (
            f"You still have {len(my_incomplete)} incomplete task(s):\n"
            f"{task_lines}\n\n"
            "Do NOT stop. Check the task list with TeamTaskList and continue "
            "working on your tasks. If a task is blocked, wait and check again. "
            "If a task is pending, pick it up with TeamTaskUpdate status='in_progress'."
        )

    def cancel(self) -> None:
        """Signal the teammate to stop."""
        self._cancel_event.set()
        if self._task is not None:
            self._task.cancel()
