"""Team strategy implementations.

These wrap the core primitives (TeamManager, SharedTaskList, MailboxRouter)
and expose them through the strategy interfaces defined in coordination.protocols.
"""

from __future__ import annotations

from typing import Any

from agent_host.teams.models import TeamConfig
from agent_host.teams.team_manager import TeamManager
from agent_host.teams.tools import (
    ALL_TEAM_TOOL_NAMES,
    LEAD_TOOLS,
    SHARED_TOOLS,
)


class TeamCoordinator:
    """Implements CoordinationStrategy by wrapping TeamManager."""

    def __init__(self, lead_session_id: str = "", config: TeamConfig | None = None) -> None:
        self._lead_session_id = lead_session_id
        self._config = config or TeamConfig()
        self._manager: TeamManager | None = None

    @property
    def manager(self) -> TeamManager | None:
        return self._manager

    @property
    def is_team_active(self) -> bool:
        return self._manager is not None

    async def on_session_start(self, session_id: str, _config: dict[str, Any]) -> None:
        self._lead_session_id = session_id

    async def on_session_shutdown(self) -> None:
        if self._manager is not None:
            await self._manager.shutdown_all()
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
        return {
            "name": info.name,
            "role": info.role,
            "status": info.status,
            "initial_prompt": initial_prompt,
        }

    async def shutdown_agent(self, name: str) -> None:
        if self._manager is not None:
            await self._manager.shutdown_teammate(name)

    async def shutdown_all(self) -> None:
        if self._manager is not None:
            await self._manager.shutdown_all()

    def get_active_agents(self) -> list[dict[str, Any]]:
        if self._manager is None:
            return []
        return [
            {"name": m.name, "role": m.role, "status": m.status, "budget": m.budget}
            for m in self._manager.get_active_agents()
        ]


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
            task = await manager.task_list.update_status(task_id, status, result=result)
        except (KeyError, ValueError) as e:
            return {"status": "error", "message": str(e)}
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
        return {"status": "success", "message": f"Message sent to {to}."}


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

        # Restore tasks
        from agent_host.teams.models import TeamTask

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
