"""Team strategy implementations.

These wrap the core primitives (TeamManager, SharedTaskList, MailboxRouter)
and expose them through the strategy interfaces defined in coordination.protocols.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import structlog

from agent_host.teams.models import TeamConfig, TeamTask
from agent_host.teams.team_manager import TeamManager
from agent_host.teams.tools import (
    ALL_TEAM_TOOL_NAMES,
    LEAD_TOOLS,
    SHARED_TOOLS,
)

if TYPE_CHECKING:
    from agent_host.budget.token_budget import TokenBudget
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.llm.client import LLMClient
    from agent_host.loop.models import LoopResult
    from agent_host.policy.policy_enforcer import PolicyEnforcer
    from agent_host.session.workspace_client import WorkspaceClient
    from agent_host.teams.teammate_session import TeammateSessionManager
    from tool_runtime import ToolRouter

logger = structlog.get_logger()

# Teammate shutdown timeout (seconds)
_SHUTDOWN_TIMEOUT = 60

# Default idle timeout for teammates (seconds)
_DEFAULT_IDLE_TIMEOUT = 300  # 5 minutes


class TeamCoordinator:
    """Implements CoordinationStrategy by wrapping TeamManager.

    Call ``set_shared_resources()`` after session creation to provide
    the resources teammates need (LLMClient, PolicyEnforcer, etc.).
    """

    def __init__(self, lead_session_id: str = "", config: TeamConfig | None = None) -> None:
        self._lead_session_id = lead_session_id
        self._config = config or TeamConfig()
        self._manager: TeamManager | None = None

        # Shared resources (set by SessionManager after session creation)
        self._llm_client: LLMClient | None = None
        self._policy_enforcer: PolicyEnforcer | None = None
        self._tool_router: ToolRouter | None = None
        self._workspace_dir: str | None = None
        self._event_emitter: EventEmitter | None = None
        self._max_context_tokens: int = 100_000
        self._workspace_client: WorkspaceClient | None = None
        self._workspace_id: str | None = None

        # Running teammate sessions and their asyncio tasks
        self._teammate_sessions: dict[str, TeammateSessionManager] = {}
        self._teammate_tasks: dict[str, asyncio.Task[LoopResult | None]] = {}

        # Wake event — signals the lead agent to resume from WaitForTeam
        self._wake_event = asyncio.Event()

        # Back-references to strategy siblings (set by TeamToolProvider/TeamContextInjector)
        self._tool_provider: TeamToolProvider | None = None
        self._context_injector: TeamContextInjector | None = None

        # Budget reallocation: lead's token budget (set via set_shared_resources)
        self._lead_token_budget: TokenBudget | None = None

        # Idle timeout tracking: per-teammate last-activity timestamp
        self._idle_timeout = _DEFAULT_IDLE_TIMEOUT
        self._last_activity: dict[str, float] = {}
        self._idle_monitor_task: asyncio.Task[None] | None = None

        # Per-teammate wake events — set when a message arrives or task unblocks
        self._teammate_wake_events: dict[str, asyncio.Event] = {}

    def set_shared_resources(
        self,
        llm_client: LLMClient,
        policy_enforcer: PolicyEnforcer,
        tool_router: ToolRouter,
        workspace_dir: str | None = None,
        event_emitter: EventEmitter | None = None,
        max_context_tokens: int = 100_000,
        workspace_client: WorkspaceClient | None = None,
        workspace_id: str | None = None,
        lead_token_budget: TokenBudget | None = None,
    ) -> None:
        """Provide shared resources from the lead's SessionManager."""
        self._llm_client = llm_client
        self._policy_enforcer = policy_enforcer
        self._tool_router = tool_router
        self._workspace_dir = workspace_dir
        self._event_emitter = event_emitter
        self._max_context_tokens = max_context_tokens
        self._workspace_client = workspace_client
        self._workspace_id = workspace_id
        self._lead_token_budget = lead_token_budget

    @property
    def manager(self) -> TeamManager | None:
        return self._manager

    @property
    def team_id(self) -> str:
        """Return the active team's ID (empty string if no team)."""
        return self._manager.team_id if self._manager else ""

    @property
    def is_team_active(self) -> bool:
        return self._manager is not None

    async def on_session_start(self, session_id: str, _config: dict[str, Any]) -> None:
        self._lead_session_id = session_id

    async def on_session_shutdown(self) -> None:
        if self._manager is not None:
            # Stop idle monitor first
            if self._idle_monitor_task and not self._idle_monitor_task.done():
                self._idle_monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._idle_monitor_task
                self._idle_monitor_task = None
            await self._shutdown_all_teammates()
            await self._upload_team_summary()
            self._manager = None

    def create_team(self, name: str, description: str = "") -> TeamManager:
        """Create and activate the team. Called by TeamToolProvider."""
        if self._manager is not None:
            msg = "Team already active"
            raise ValueError(msg)
        config = TeamConfig(
            name=name, description=description, max_teammates=self._config.max_teammates
        )
        self._manager = TeamManager(lead_session_id=self._lead_session_id, config=config)
        if self._event_emitter:
            self._event_emitter.emit_team_created(self._manager.team_id, name)
        return self._manager

    async def spawn_agent(
        self,
        name: str,
        role: str,
        initial_prompt: str,
        budget: int,
    ) -> dict[str, Any]:
        if self._manager is None:
            msg = "No active team — call CreateTeam first"
            raise RuntimeError(msg)
        info = await self._manager.create_teammate(name, role, budget)

        # If shared resources are available, create and launch the teammate loop
        if self._llm_client and self._policy_enforcer and self._tool_router:
            from agent_host.teams.teammate_session import TeammateSessionManager

            wake_event = asyncio.Event()
            self._teammate_wake_events[name] = wake_event
            teammate = TeammateSessionManager(
                name=name,
                role=role,
                team_name=self._manager.config.name,
                team_id=self._manager.team_id,
                llm_client=self._llm_client,
                policy_enforcer=self._policy_enforcer,
                tool_router=self._tool_router,
                workspace_dir=self._workspace_dir,
                tool_provider=self._tool_provider or TeamToolProvider(self),
                context_injector=self._context_injector or TeamContextInjector(self),
                budget=budget or 100_000,
                max_context_tokens=self._max_context_tokens,
                event_emitter=self._event_emitter,
                workspace_client=self._workspace_client,
                workspace_id=self._workspace_id,
                on_activity=self.record_teammate_activity,
                task_list=self._manager.task_list,
                wake_event=wake_event,
            )
            self._teammate_sessions[name] = teammate
            task = asyncio.create_task(teammate.run(initial_prompt), name=f"teammate-{name}")
            task.add_done_callback(
                lambda _t: self._on_teammate_done(_t.get_name().removeprefix("teammate-"))
            )
            self._teammate_tasks[name] = task
            info.status = "running"
            self._last_activity[name] = time.monotonic()

            # Start the idle monitor if not already running
            if self._idle_monitor_task is None or self._idle_monitor_task.done():
                self._idle_monitor_task = asyncio.create_task(
                    self._idle_monitor_loop(), name="team-idle-monitor"
                )

        if self._event_emitter and self._manager:
            self._event_emitter.emit_teammate_created(self._manager.team_id, name, role)

        return {
            "name": info.name,
            "role": info.role,
            "initial_prompt": initial_prompt,
        }

    async def shutdown_agent(self, name: str) -> None:
        if self._manager is not None:
            await self._shutdown_teammate(name)

    async def shutdown_all(self) -> None:
        if self._manager is not None:
            await self._shutdown_all_teammates()

    def get_active_agents(self) -> list[dict[str, Any]]:
        if self._manager is None:
            return []
        return [
            {"name": m.name, "role": m.role, "status": m.status, "budget": m.budget}
            for m in self._manager.get_active_agents()
        ]

    async def resume_teammates(self) -> None:
        """Re-spawn teammate loops after crash recovery.

        Called after ``TeamCheckpointProvider.restore()`` has rebuilt the
        TeamManager with member metadata and task list.  Only members whose
        status was ``"running"`` are resumed.
        """
        if self._manager is None:
            return
        if not (self._llm_client and self._policy_enforcer and self._tool_router):
            logger.warning(
                "resume_teammates_skipped",
                reason="shared_resources_not_set",
                team_id=self.team_id,
            )
            return

        from agent_host.teams.teammate_session import TeammateSessionManager

        for name, info in list(self._manager.members.items()):
            if info.status != "running":
                continue

            # Build a resume prompt — teammate will see task list via context injection
            resume_prompt = (
                "You are resuming after a restart. Check the team task list "
                "for your assigned tasks and continue working."
            )
            wake_event = asyncio.Event()
            self._teammate_wake_events[name] = wake_event
            teammate = TeammateSessionManager(
                name=name,
                role=info.role,
                team_name=self._manager.config.name,
                team_id=self._manager.team_id,
                llm_client=self._llm_client,
                policy_enforcer=self._policy_enforcer,
                tool_router=self._tool_router,
                workspace_dir=self._workspace_dir,
                tool_provider=self._tool_provider or TeamToolProvider(self),
                context_injector=self._context_injector or TeamContextInjector(self),
                budget=info.budget or 100_000,
                max_context_tokens=self._max_context_tokens,
                event_emitter=self._event_emitter,
                workspace_client=self._workspace_client,
                workspace_id=self._workspace_id,
                on_activity=self.record_teammate_activity,
                task_list=self._manager.task_list,
                wake_event=wake_event,
            )
            self._teammate_sessions[name] = teammate
            task = asyncio.create_task(teammate.run(resume_prompt), name=f"teammate-{name}")
            task.add_done_callback(
                lambda _t: self._on_teammate_done(_t.get_name().removeprefix("teammate-"))
            )
            self._teammate_tasks[name] = task
            self._last_activity[name] = time.monotonic()
            logger.info("teammate_resumed", name=name, role=info.role, team_id=self.team_id)

    def wake(self) -> None:
        """Signal the lead to wake from WaitForTeam."""
        self._wake_event.set()

    def wake_teammate(self, name: str) -> None:
        """Signal a specific teammate to wake from a blocked wait."""
        event = self._teammate_wake_events.get(name)
        if event is not None:
            event.set()

    async def wait_for_wake(self, timeout: float = 120.0) -> str:
        """Block until a wake condition fires or timeout. Returns reason.

        If a wake signal arrived before this call, returns immediately
        (avoids lost-wakeup race condition).
        """
        if self._wake_event.is_set():
            self._wake_event.clear()
            return "event"
        self._wake_event.clear()
        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
            return "event"
        except TimeoutError:
            return "timeout"

    def record_teammate_activity(self, name: str) -> None:
        """Record that a teammate performed an action (resets idle timer)."""
        if name in self._last_activity:
            self._last_activity[name] = time.monotonic()

    async def _idle_monitor_loop(self, check_interval: float = 60.0) -> None:
        """Background task that checks for idle teammates periodically."""
        try:
            while self._manager is not None and self._last_activity:
                await asyncio.sleep(check_interval)
                now = time.monotonic()
                idle_names = [
                    name
                    for name, last in self._last_activity.items()
                    if now - last > self._idle_timeout
                    and name in self._teammate_tasks
                    and not self._teammate_tasks[name].done()
                ]
                for name in idle_names:
                    logger.warning(
                        "teammate_idle_timeout",
                        name=name,
                        team_id=self.team_id,
                        idle_seconds=now - self._last_activity.get(name, now),
                    )
                    await self._shutdown_teammate(name)
        except asyncio.CancelledError:
            pass

    def _on_teammate_done(self, name: str) -> None:
        """Called when a teammate's asyncio task finishes (naturally or via error)."""
        # Reclaim unused budget from the finished teammate
        reclaimed = 0
        session = self._teammate_sessions.get(name)
        if session is not None and self._lead_token_budget is not None:
            reclaimed = session._token_budget.remaining
            if reclaimed > 0:
                self._lead_token_budget.add_budget(reclaimed)

        logger.info(
            "teammate_task_done",
            name=name,
            team_id=self.team_id,
            reclaimed_tokens=reclaimed,
        )

        # Update member status to "stopped"
        if self._manager and name in self._manager.members:
            self._manager.members[name].status = "stopped"

        # Clean up idle tracking and wake event
        self._last_activity.pop(name, None)
        self._teammate_wake_events.pop(name, None)

        # Notify UI that teammate is done
        if self._event_emitter and self._manager:
            self._event_emitter.emit_teammate_removed(self._manager.team_id, name)

        self._wake_event.set()

    async def _upload_team_summary(self) -> None:
        """Upload a JSON summary of the team run to Workspace Service (best-effort)."""
        if not self._workspace_client or not self._workspace_id or not self._manager:
            return
        try:
            import json

            tasks = await self._manager.task_list.list_tasks()
            summary = {
                "team_id": self._manager.team_id,
                "team_name": self._manager.config.name,
                "lead_session_id": self._lead_session_id,
                "teammates": [
                    {
                        "name": s.name,
                        "role": s.role,
                        "session_id": s._session_id,
                    }
                    for s in self._teammate_sessions.values()
                ],
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "title": t.title,
                        "status": t.status,
                        "assignee": t.assignee,
                        "result": t.result,
                    }
                    for t in tasks
                ],
            }
            await self._workspace_client.upload_artifact(
                workspace_id=self._workspace_id,
                session_id=self._lead_session_id,
                artifact_data=json.dumps(summary, indent=2).encode(),
                artifact_type="team_summary",
                artifact_name=f"team-summary-{self._manager.team_id}",
                content_type="application/json",
            )
            logger.info("team_summary_uploaded", team_id=self._manager.team_id)
        except Exception:
            logger.warning(
                "team_summary_upload_failed", team_id=self._manager.team_id, exc_info=True
            )

    async def _shutdown_teammate(self, name: str) -> None:
        """Cancel a teammate's loop and clean up."""
        session = self._teammate_sessions.pop(name, None)
        task = self._teammate_tasks.pop(name, None)

        if session is not None:
            session.cancel()

        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=_SHUTDOWN_TIMEOUT)
            except TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            except asyncio.CancelledError:
                pass

        if self._manager is not None:
            if self._event_emitter:
                self._event_emitter.emit_teammate_removed(self._manager.team_id, name)
            with contextlib.suppress(KeyError):
                await self._manager.shutdown_teammate(name)

    async def _shutdown_all_teammates(self) -> None:
        """Shut down all teammate loops concurrently."""
        names = list(self._teammate_sessions.keys())
        if names:
            await asyncio.gather(
                *(self._shutdown_teammate(n) for n in names),
                return_exceptions=True,
            )
        # Clean up any remaining members in manager
        if self._manager is not None:
            remaining = list(self._manager.members.keys())
            for name in remaining:
                with contextlib.suppress(KeyError):
                    await self._manager.shutdown_teammate(name)


class TeamToolProvider:
    """Implements ToolProviderStrategy — exposes team tools filtered by role."""

    def __init__(self, coordinator: TeamCoordinator) -> None:
        self._coordinator = coordinator

    def get_tool_definitions(self, agent_role: str) -> list[dict[str, Any]]:
        if not self._coordinator.is_team_active and agent_role == "lead":
            # Before team is created, only expose CreateTeam
            return [t for t in LEAD_TOOLS if t["function"]["name"] == "CreateTeam"]
        if not self._coordinator.is_team_active:
            return []
        if agent_role == "lead":
            return LEAD_TOOLS + SHARED_TOOLS
        if agent_role == "teammate":
            return list(SHARED_TOOLS)
        return []

    async def handle_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_name: str,
    ) -> dict[str, Any]:
        if tool_name == "CreateTeam":
            return self._handle_create_team(arguments)
        if tool_name == "CreateTeammate":
            return await self._handle_create_teammate(arguments)
        if tool_name == "ShutdownTeammate":
            return await self._handle_shutdown_teammate(arguments)
        if tool_name == "ShutdownTeam":
            return await self._handle_shutdown_team()
        if tool_name == "TeamTaskCreate":
            return await self._handle_task_create(arguments, agent_name)
        if tool_name == "TeamTaskUpdate":
            return await self._handle_task_update(arguments)
        if tool_name == "TeamTaskList":
            return await self._handle_task_list(arguments)
        if tool_name == "SendTeamMessage":
            return await self._handle_send_message(arguments, agent_name)
        if tool_name == "WaitForTeam":
            return await self._handle_wait_for_team(arguments)
        msg = f"TeamToolProvider does not handle tool: {tool_name}"
        raise RuntimeError(msg)

    def owns_tool(self, tool_name: str) -> bool:
        return tool_name in ALL_TEAM_TOOL_NAMES

    # ── Tool handlers ──────────────────────────────────────────

    def _handle_create_team(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name = arguments.get("name", "")
        if not name:
            return {"status": "error", "message": "name is required"}
        description = arguments.get("description", "")
        try:
            manager = self._coordinator.create_team(name, description)
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        return {
            "status": "success",
            "team_id": manager.team_id,
            "message": f"Team '{name}' created. Use CreateTeammate to add agents.",
        }

    async def _handle_create_teammate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name = arguments.get("name", "")
        role = arguments.get("role", "")
        initial_prompt = arguments.get("initial_prompt", "")
        if not name or not role or not initial_prompt:
            return {"status": "error", "message": "name, role, and initial_prompt are required"}
        try:
            result = await self._coordinator.spawn_agent(
                name=name,
                role=role,
                initial_prompt=initial_prompt,
                budget=0,
            )
        except (RuntimeError, ValueError) as e:
            return {"status": "error", "message": str(e)}
        return {
            "status": "success",
            "name": result["name"],
            "role": result["role"],
            "initial_prompt": result["initial_prompt"],
        }

    async def _handle_shutdown_teammate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name = arguments.get("name", "")
        if not name:
            return {"status": "error", "message": "name is required"}
        try:
            await self._coordinator.shutdown_agent(name)
        except KeyError as e:
            return {"status": "error", "message": str(e)}
        return {"status": "success", "message": f"Teammate '{name}' shut down."}

    async def _handle_shutdown_team(self) -> dict[str, Any]:
        await self._coordinator.shutdown_all()
        return {"status": "success", "message": "All teammates shut down."}

    async def _handle_task_create(
        self, arguments: dict[str, Any], agent_name: str
    ) -> dict[str, Any]:
        manager = self._coordinator.manager
        if manager is None:
            return {"status": "error", "message": "No active team"}
        title = arguments.get("title", "")
        description = arguments.get("description", "")
        if not title:
            return {"status": "error", "message": "title is required"}
        blocked_by = arguments.get("blocked_by")
        try:
            task = await manager.task_list.create_task(
                title=title,
                description=description,
                blocked_by=blocked_by,
                created_by=agent_name,
            )
        except KeyError as e:
            return {"status": "error", "message": str(e)}
        # Emit task_updated notification
        if self._coordinator._event_emitter and self._coordinator._manager:
            self._coordinator._event_emitter.emit_team_task_updated(
                self._coordinator._manager.team_id,
                _task_to_dict(task),
            )
        # Wake the lead when a teammate creates a task
        if agent_name != "lead":
            self._coordinator.wake()
        return {"status": "success", "task_id": task.task_id, "title": task.title}

    async def _handle_task_update(self, arguments: dict[str, Any]) -> dict[str, Any]:
        manager = self._coordinator.manager
        if manager is None:
            return {"status": "error", "message": "No active team"}
        task_id = arguments.get("task_id", "")
        status = arguments.get("status", "")
        if not task_id or not status:
            return {"status": "error", "message": "task_id and status are required"}
        result = arguments.get("result")
        try:
            task, unblocked = await manager.task_list.update_status(
                task_id, status, result=result
            )
        except (KeyError, ValueError) as e:
            return {"status": "error", "message": str(e)}
        # Emit task_updated notification
        if self._coordinator._event_emitter and self._coordinator._manager:
            self._coordinator._event_emitter.emit_team_task_updated(
                self._coordinator._manager.team_id,
                _task_to_dict(task),
            )
            # Also emit updates for newly unblocked tasks
            for unblocked_task in unblocked:
                self._coordinator._event_emitter.emit_team_task_updated(
                    self._coordinator._manager.team_id,
                    _task_to_dict(unblocked_task),
                )
        # Wake teammates whose tasks just became unblocked
        for unblocked_task in unblocked:
            creator = unblocked_task.created_by
            if creator:
                self._coordinator.wake_teammate(creator)
            assignee = unblocked_task.assignee
            if assignee and assignee != creator:
                self._coordinator.wake_teammate(assignee)
        # Wake the lead when a task completes or fails
        if status in ("completed", "failed"):
            self._coordinator.wake()
        return {"status": "success", "task_id": task.task_id, "new_status": task.status}

    async def _handle_task_list(self, arguments: dict[str, Any]) -> dict[str, Any]:
        manager = self._coordinator.manager
        if manager is None:
            return {"status": "error", "message": "No active team"}
        tasks = await manager.task_list.list_tasks(
            status=arguments.get("status"),
            assignee=arguments.get("assignee"),
        )
        return {
            "status": "success",
            "tasks": [
                {
                    "task_id": t.task_id,
                    "title": t.title,
                    "status": t.status,
                    "assignee": t.assignee,
                    "result": t.result,
                }
                for t in tasks
            ],
        }

    async def _handle_send_message(
        self, arguments: dict[str, Any], agent_name: str
    ) -> dict[str, Any]:
        manager = self._coordinator.manager
        if manager is None:
            return {"status": "error", "message": "No active team"}
        to = arguments.get("to", "")
        content = arguments.get("content", "")
        if not to or not content:
            return {"status": "error", "message": "to and content are required"}
        try:
            if to == "all":
                await manager.broadcast_message(agent_name, content)
            else:
                await manager.send_message(agent_name, to, content)
        except KeyError as e:
            return {"status": "error", "message": str(e)}
        # Emit team/message notification
        if self._coordinator._event_emitter and self._coordinator._manager:
            self._coordinator._event_emitter.emit_team_message(
                self._coordinator._manager.team_id,
                from_agent=agent_name,
                to_agent=to,
                content=content,
            )
        # Wake the lead when a message is sent to them
        if to == "lead" or to == "all":
            self._coordinator.wake()
        # Wake teammate if they're waiting on blocked tasks
        if to == "all":
            for member_name in (manager.members if manager else {}):
                if member_name != agent_name:
                    self._coordinator.wake_teammate(member_name)
        elif to != "lead":
            self._coordinator.wake_teammate(to)
        return {"status": "success", "message": f"Message sent to {to}."}

    async def _handle_wait_for_team(self, arguments: dict[str, Any]) -> dict[str, Any]:
        timeout = float(arguments.get("timeout", 120))
        reason = await self._coordinator.wait_for_wake(timeout=timeout)

        # Build a status snapshot for the lead
        manager = self._coordinator.manager
        result: dict[str, Any] = {"status": "success", "wake_reason": reason}
        if manager is not None:
            tasks = await manager.task_list.list_tasks()
            completed = sum(1 for t in tasks if t.status == "completed")
            failed = sum(1 for t in tasks if t.status == "failed")
            in_progress = sum(1 for t in tasks if t.status == "in_progress")
            pending = sum(1 for t in tasks if t.status in ("pending", "blocked"))
            result["task_summary"] = {
                "total": len(tasks),
                "completed": completed,
                "failed": failed,
                "in_progress": in_progress,
                "pending": pending,
            }
            result["all_tasks_done"] = (completed + failed) == len(tasks) and len(tasks) > 0

            # Include teammate statuses
            agents = self._coordinator.get_active_agents()
            result["teammates"] = [{"name": a["name"], "status": a["status"]} for a in agents]

        return result


def _task_to_dict(task: TeamTask) -> dict[str, Any]:
    """Convert a TeamTask to a dict for the team/task_updated notification."""
    return {
        "task_id": task.task_id,
        "title": task.title,
        "description": task.description,
        "status": task.status,
        "assignee": task.assignee,
        "created_by": task.created_by,
        "result": task.result,
        "blocked_by": task.blocked_by,
    }


class TeamContextInjector:
    """Implements ContextInjectionStrategy — injects messages + task status."""

    OVERHEAD_TOKENS = 200  # conservative estimate per injection turn

    def __init__(self, coordinator: TeamCoordinator) -> None:
        self._coordinator = coordinator

    async def get_injections(self, agent_name: str) -> list[str]:
        manager = self._coordinator.manager
        if manager is None:
            return []

        parts: list[str] = []

        # Inject pending messages
        messages = await manager.mailbox.poll(agent_name)
        if messages:
            lines = [f"From @{m.from_agent}: {m.content!r}" for m in messages]
            parts.append("[Team Messages]\n" + "\n".join(lines))

        # Inject current task summary
        tasks = await manager.task_list.list_tasks()
        if tasks:
            summary_lines: list[str] = []
            for t in tasks:
                assignee = f", assigned to {t.assignee}" if t.assignee else ""
                summary_lines.append(f"- [{t.status}] {t.title} (id={t.task_id}{assignee})")
            parts.append("[Team Tasks]\n" + "\n".join(summary_lines))

        return parts

    def estimate_overhead_tokens(self) -> int:
        if self._coordinator.is_team_active:
            return self.OVERHEAD_TOKENS
        return 0


class TeamCheckpointProvider:
    """Implements CheckpointStrategy — captures/restores coordination state."""

    def __init__(self, coordinator: TeamCoordinator) -> None:
        self._coordinator = coordinator

    def capture(self) -> dict[str, Any]:
        manager = self._coordinator.manager
        if manager is None:
            return {}
        return {
            "team_id": manager.team_id,
            "config": {
                "name": manager.config.name,
                "description": manager.config.description,
                "max_teammates": manager.config.max_teammates,
            },
            "members": {
                name: {
                    "role": info.role,
                    "budget": info.budget,
                    "status": info.status,
                    "session_id": info.session_id,
                }
                for name, info in manager.members.items()
            },
            "tasks": [
                {
                    "task_id": t.task_id,
                    "title": t.title,
                    "description": t.description,
                    "status": t.status,
                    "assignee": t.assignee,
                    "created_by": t.created_by,
                    "blocked_by": t.blocked_by,
                    "result": t.result,
                }
                for t in manager.task_list._tasks.values()
            ],
        }

    async def restore(self, state: dict[str, Any]) -> None:
        if not state:
            return

        config = state.get("config", {})
        team_config = TeamConfig(
            name=config.get("name", ""),
            description=config.get("description", ""),
            max_teammates=config.get("max_teammates", 5),
        )
        manager = self._coordinator.create_team(team_config.name, team_config.description)
        manager.config = team_config

        # Restore members
        for name, info in state.get("members", {}).items():
            teammate = await manager.create_teammate(name, info["role"], info["budget"])
            teammate.status = info.get("status", "created")
            teammate.session_id = info.get("session_id")

        # Restore tasks (acquire lock for consistency even during startup)
        from agent_host.teams.models import TeamTask

        async with manager.task_list._lock:
            for t in state.get("tasks", []):
                task = TeamTask(
                    task_id=t["task_id"],
                    title=t.get("title", ""),
                    description=t.get("description", ""),
                    status=t.get("status", "pending"),
                    assignee=t.get("assignee"),
                    created_by=t.get("created_by", ""),
                    blocked_by=t.get("blocked_by", []),
                    result=t.get("result"),
                )
                manager.task_list._tasks[task.task_id] = task

        # Re-spawn teammate loops for members that were running
        await self._coordinator.resume_teammates()
