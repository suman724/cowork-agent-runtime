"""AgentToolHandler — internal tools for managing agent working memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent_host.memory.memory_manager import MemoryManager
    from agent_host.memory.working_memory import WorkingMemory
    from agent_host.skills.models import SkillDefinition

# Agent-internal tool names — these bypass PolicyEnforcer and ToolRouter
AGENT_TOOL_NAMES = {
    "TaskTracker",
    "CreatePlan",
    "SpawnAgent",
    "SaveMemory",
    "RecallMemory",
    "ListMemories",
    "DeleteMemory",
}


class AgentToolHandler:
    """Handles agent-internal tool calls (TaskTracker, CreatePlan).

    These tools manipulate the agent's working memory and do NOT
    go through the PolicyEnforcer or ToolRouter.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        skills: list[SkillDefinition] | None = None,
        memory_manager: MemoryManager | None = None,
        spawn_sub_agent: Callable[..., Awaitable[dict[str, Any]]] | None = None,
        execute_skill: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self._working_memory = working_memory
        self._skills = {s.name: s for s in (skills or [])}
        self._skill_tool_names = {f"Skill_{s.name}" for s in (skills or [])}
        self._memory_manager = memory_manager
        self._spawn_sub_agent = spawn_sub_agent
        self._execute_skill = execute_skill

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
        if tool_name == "SaveMemory":
            return await self._handle_save_memory(arguments)
        if tool_name == "RecallMemory":
            return await self._handle_recall_memory(arguments)
        if tool_name == "ListMemories":
            return await self._handle_list_memories()
        if tool_name == "DeleteMemory":
            return await self._handle_delete_memory(arguments)
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
        # Add memory tools if memory_manager is configured
        if self._memory_manager:
            defs.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "SaveMemory",
                            "description": (
                                "Save information to persistent memory that"
                                " survives across sessions. Write to"
                                " MEMORY.md (concise index, loaded every"
                                " session) or topic files (e.g.,"
                                " 'debugging.md') for detailed notes."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file": {
                                        "type": "string",
                                        "description": (
                                            "Memory filename (default: MEMORY.md). "
                                            "Must be [a-zA-Z0-9_-]+.md"
                                        ),
                                        "default": "MEMORY.md",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The full content to write to the file.",
                                    },
                                },
                                "required": ["content"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "RecallMemory",
                            "description": (
                                "Read a specific memory topic file. "
                                "Use ListMemories first to see available files."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file": {
                                        "type": "string",
                                        "description": (
                                            "The memory filename to read (e.g., 'debugging.md')."
                                        ),
                                    },
                                },
                                "required": ["file"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "ListMemories",
                            "description": (
                                "List all available persistent memory files with their sizes."
                            ),
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "DeleteMemory",
                            "description": (
                                "Delete a persistent memory topic file. Cannot delete MEMORY.md."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file": {
                                        "type": "string",
                                        "description": (
                                            "The memory filename to delete (e.g., 'old-notes.md')."
                                        ),
                                    },
                                },
                                "required": ["file"],
                            },
                        },
                    },
                ]
            )

        # Add SpawnAgent if sub-agent spawning is available
        if self._spawn_sub_agent:
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
        # Add skill tools (skip skills with disable_model_invocation=True)
        for skill in self._skills.values():
            if skill.disable_model_invocation:
                continue  # LLM cannot see or auto-trigger this skill
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
        if not self._spawn_sub_agent:
            return {"status": "error", "message": "Sub-agent spawning is not available"}

        task = arguments.get("task", "")
        if not task:
            return {"status": "error", "message": "task is required"}
        context = arguments.get("context", "")
        return await self._spawn_sub_agent(
            task=task,
            context=context,
            parent_task_id="",
        )

    async def _handle_save_memory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle SaveMemory tool calls."""
        if not self._memory_manager:
            return {"status": "error", "message": "Persistent memory is not available"}
        return await self._memory_manager.handle_save_memory(arguments)

    async def _handle_recall_memory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle RecallMemory tool calls."""
        if not self._memory_manager:
            return {"status": "error", "message": "Persistent memory is not available"}
        return await self._memory_manager.handle_recall_memory(arguments)

    async def _handle_list_memories(self) -> dict[str, Any]:
        """Handle ListMemories tool calls."""
        if not self._memory_manager:
            return {"status": "error", "message": "Persistent memory is not available"}
        return await self._memory_manager.handle_list_memories()

    async def _handle_delete_memory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle DeleteMemory tool calls."""
        if not self._memory_manager:
            return {"status": "error", "message": "Persistent memory is not available"}
        return await self._memory_manager.handle_delete_memory(arguments)

    async def _handle_skill(
        self, tool_name: str, arguments: dict[str, Any], task_id: str
    ) -> dict[str, Any]:
        """Handle skill tool calls by delegating to LoopRuntime."""
        skill_name = tool_name.removeprefix("Skill_")
        skill = self._skills.get(skill_name)
        if not skill:
            return {"status": "error", "message": f"Unknown skill: {skill_name}"}

        if not self._execute_skill:
            return {"status": "error", "message": "Skill execution is not available"}

        return await self._execute_skill(
            skill=skill,
            arguments=arguments,
            parent_task_id=task_id,
        )
