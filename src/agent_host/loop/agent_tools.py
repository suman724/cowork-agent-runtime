"""AgentToolHandler — internal tools for managing agent working memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_host.loop.sub_agent import SubAgentManager
    from agent_host.memory.working_memory import WorkingMemory
    from agent_host.skills.models import SkillDefinition
    from agent_host.skills.skill_executor import SkillExecutor

# Agent-internal tool names — these bypass PolicyEnforcer and ToolRouter
AGENT_TOOL_NAMES = {"TaskTracker", "CreatePlan", "SpawnAgent"}


class AgentToolHandler:
    """Handles agent-internal tool calls (TaskTracker, CreatePlan).

    These tools manipulate the agent's working memory and do NOT
    go through the PolicyEnforcer or ToolRouter.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        sub_agent_manager: SubAgentManager | None = None,
        skill_executor: SkillExecutor | None = None,
        skills: list[SkillDefinition] | None = None,
    ) -> None:
        self._working_memory = working_memory
        self._sub_agent_manager = sub_agent_manager
        self._skill_executor = skill_executor
        self._skills = {s.name: s for s in (skills or [])}
        self._skill_tool_names = {f"Skill_{s.name}" for s in (skills or [])}

    def is_agent_tool(self, name: str) -> bool:
        """Check if a tool name is an agent-internal tool or a skill."""
        return name in AGENT_TOOL_NAMES or name in self._skill_tool_names

    async def execute(
        self, tool_name: str, arguments: dict[str, Any], task_id: str = ""
    ) -> dict[str, Any]:
        """Execute an agent-internal tool and return the result."""
        if tool_name == "TaskTracker":
            return self._handle_task_tracker(arguments)
        if tool_name == "CreatePlan":
            return self._handle_create_plan(arguments)
        if tool_name == "SpawnAgent":
            return await self._handle_spawn_agent(arguments)
        if tool_name in self._skill_tool_names:
            return await self._handle_skill(tool_name, arguments, task_id)
        return {"status": "error", "message": f"Unknown agent tool: {tool_name}"}

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool definitions for agent-internal tools."""
        defs: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "TaskTracker",
                    "description": (
                        "Manage your task list. Actions: 'create' (add a task), "
                        "'update' (change status/content), 'list' (show all tasks)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["create", "update", "list"],
                                "description": "The action to perform.",
                            },
                            "content": {
                                "type": "string",
                                "description": (
                                    "Task description (for 'create') or new content (for 'update')."
                                ),
                            },
                            "taskId": {
                                "type": "string",
                                "description": "Task ID (required for 'update').",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "failed"],
                                "description": "New status (for 'update').",
                            },
                        },
                        "required": ["action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "CreatePlan",
                    "description": (
                        "Create or replace the current plan with a goal and ordered steps."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "The overall goal of the plan.",
                            },
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Ordered list of step descriptions.",
                            },
                        },
                        "required": ["goal", "steps"],
                    },
                },
            },
        ]
        # Add SpawnAgent if sub_agent_manager is configured
        if self._sub_agent_manager:
            defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": "SpawnAgent",
                        "description": (
                            "Spawn a focused sub-agent to work on a specific task. "
                            "The sub-agent has its own context and runs independently."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "The task for the sub-agent.",
                                },
                                "context": {
                                    "type": "string",
                                    "description": "Relevant context from the current work.",
                                },
                            },
                            "required": ["task"],
                        },
                    },
                }
            )
        # Add skill tools
        for skill in self._skills.values():
            skill_def: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": f"Skill_{skill.name}",
                    "description": skill.description,
                },
            }
            if skill.input_schema:
                skill_def["function"]["parameters"] = skill.input_schema
            else:
                skill_def["function"]["parameters"] = {"type": "object", "properties": {}}
            defs.append(skill_def)

        return defs

    def _handle_task_tracker(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle TaskTracker tool calls."""
        action = arguments.get("action", "")
        tracker = self._working_memory.task_tracker

        if action == "create":
            content = arguments.get("content", "")
            if not content:
                return {"status": "error", "message": "content is required for create"}
            task_id = tracker.create_task(content)
            return {"status": "success", "taskId": task_id, "message": f"Created task: {content}"}

        if action == "update":
            task_id = arguments.get("taskId", "")
            if not task_id:
                return {"status": "error", "message": "taskId is required for update"}
            status = arguments.get("status")
            content = arguments.get("content")
            if tracker.update_task(task_id, status=status, content=content):
                return {"status": "success", "message": f"Updated task {task_id}"}
            return {"status": "error", "message": f"Task not found: {task_id}"}

        if action == "list":
            tasks = tracker.tasks
            return {
                "status": "success",
                "tasks": [{"id": t.id, "content": t.content, "status": t.status} for t in tasks],
            }

        return {"status": "error", "message": f"Unknown action: {action}"}

    def _handle_create_plan(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle CreatePlan tool calls."""
        from agent_host.memory.plan import Plan, PlanStep

        goal = arguments.get("goal", "")
        step_descriptions = arguments.get("steps", [])

        if not goal:
            return {"status": "error", "message": "goal is required"}
        if not step_descriptions:
            return {"status": "error", "message": "steps is required (non-empty list)"}

        steps = [PlanStep(description=desc) for desc in step_descriptions]
        self._working_memory.plan = Plan(goal=goal, steps=steps)

        return {
            "status": "success",
            "message": f"Plan created with {len(steps)} steps",
            "goal": goal,
        }

    async def _handle_spawn_agent(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle SpawnAgent tool calls."""
        if not self._sub_agent_manager:
            return {"status": "error", "message": "Sub-agent spawning is not available"}

        task = arguments.get("task", "")
        if not task:
            return {"status": "error", "message": "task is required"}

        context = arguments.get("context", "")
        return await self._sub_agent_manager.spawn(
            task=task,
            context=context,
            parent_task_id="",  # Will be set by the caller if needed
        )

    async def _handle_skill(
        self, tool_name: str, arguments: dict[str, Any], task_id: str
    ) -> dict[str, Any]:
        """Handle skill tool calls by delegating to SkillExecutor."""
        if not self._skill_executor:
            return {"status": "error", "message": "Skill execution is not available"}

        # Strip "Skill_" prefix to get the skill name
        skill_name = tool_name.removeprefix("Skill_")
        skill = self._skills.get(skill_name)
        if not skill:
            return {"status": "error", "message": f"Unknown skill: {skill_name}"}

        return await self._skill_executor.execute(
            skill=skill,
            arguments=arguments,
            parent_task_id=task_id,
        )
