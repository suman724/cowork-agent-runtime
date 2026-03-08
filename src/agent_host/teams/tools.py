"""Team tool definitions in OpenAI function-calling format."""

from __future__ import annotations

from typing import Any

# Lead-only tools
LEAD_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "CreateTeam",
            "description": (
                "Initialize a team to coordinate multiple agents working together. "
                "Call this before creating teammates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Team name (short identifier).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Team objective.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "CreateTeammate",
            "description": (
                "Spawn a new teammate agent to work on tasks. "
                "The teammate gets its own context and runs independently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short identifier (e.g. 'researcher').",
                    },
                    "role": {
                        "type": "string",
                        "description": "Role description for the teammate's context.",
                    },
                    "initial_prompt": {
                        "type": "string",
                        "description": "First instruction for the teammate.",
                    },
                },
                "required": ["name", "role", "initial_prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ShutdownTeammate",
            "description": "Gracefully stop a specific teammate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Teammate name to shut down.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ShutdownTeam",
            "description": "Shut down all teammates and clean up the team.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "WaitForTeam",
            "description": (
                "Wait for teammates to make progress. Blocks until a task is "
                "completed/failed, a message arrives for you, or a teammate finishes. "
                "Returns current team status. Use this instead of polling in a loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "number",
                        "description": "Max seconds to wait (default: 120).",
                    },
                },
            },
        },
    },
]

# Tools available to all team members (lead + teammates)
SHARED_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "TeamTaskCreate",
            "description": "Add a task to the shared team task list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title."},
                    "description": {"type": "string", "description": "Task description."},
                    "blocked_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task IDs this depends on (optional).",
                    },
                },
                "required": ["title", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TeamTaskUpdate",
            "description": (
                "Update a team task's status. Set to 'completed' with a result "
                "summary when done, or 'failed' if unable to complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to update."},
                    "status": {
                        "type": "string",
                        "enum": ["in_progress", "completed", "failed"],
                        "description": "New status.",
                    },
                    "result": {
                        "type": "string",
                        "description": "Completion summary (when status=completed).",
                    },
                },
                "required": ["task_id", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TeamTaskList",
            "description": "View all tasks in the shared team task list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status (optional).",
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Filter by assignee (optional).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "SendTeamMessage",
            "description": "Send a message to a teammate or broadcast to all.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Teammate name, or 'all' for broadcast.",
                    },
                    "content": {"type": "string", "description": "Message content."},
                },
                "required": ["to", "content"],
            },
        },
    },
]

# All tool names for ownership checks
LEAD_TOOL_NAMES = {t["function"]["name"] for t in LEAD_TOOLS}
SHARED_TOOL_NAMES = {t["function"]["name"] for t in SHARED_TOOLS}
ALL_TEAM_TOOL_NAMES = LEAD_TOOL_NAMES | SHARED_TOOL_NAMES
