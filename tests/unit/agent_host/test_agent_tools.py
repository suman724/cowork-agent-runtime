"""Tests for AgentToolHandler — internal tools for working memory."""

from __future__ import annotations

from agent_host.loop.agent_tools import AgentToolHandler
from agent_host.memory.working_memory import WorkingMemory


class TestAgentToolHandlerRouting:
    def test_is_agent_tool(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        assert handler.is_agent_tool("TaskTracker")
        assert handler.is_agent_tool("CreatePlan")
        assert not handler.is_agent_tool("ReadFile")
        assert not handler.is_agent_tool("WriteFile")

    def test_get_tool_definitions(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        defs = handler.get_tool_definitions()
        names = {d["function"]["name"] for d in defs}
        assert "TaskTracker" in names
        assert "CreatePlan" in names
        for d in defs:
            assert d["type"] == "function"
            assert "parameters" in d["function"]

    async def test_unknown_tool(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        result = await handler.execute("UnknownTool", {})
        assert result["status"] == "error"


class TestTaskTrackerTool:
    async def test_create_task(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        result = await handler.execute("TaskTracker", {"action": "create", "content": "Do X"})
        assert result["status"] == "success"
        assert "taskId" in result
        assert len(wm.task_tracker.tasks) == 1

    async def test_create_task_missing_content(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        result = await handler.execute("TaskTracker", {"action": "create"})
        assert result["status"] == "error"

    async def test_update_task(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        create_result = await handler.execute(
            "TaskTracker", {"action": "create", "content": "Task A"}
        )
        task_id = create_result["taskId"]

        update_result = await handler.execute(
            "TaskTracker", {"action": "update", "taskId": task_id, "status": "completed"}
        )
        assert update_result["status"] == "success"
        assert wm.task_tracker.get_task(task_id).status == "completed"  # type: ignore[union-attr]

    async def test_update_nonexistent_task(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        result = await handler.execute(
            "TaskTracker", {"action": "update", "taskId": "bad-id", "status": "completed"}
        )
        assert result["status"] == "error"

    async def test_update_missing_task_id(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        result = await handler.execute("TaskTracker", {"action": "update", "status": "completed"})
        assert result["status"] == "error"

    async def test_list_tasks(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        await handler.execute("TaskTracker", {"action": "create", "content": "Task A"})
        await handler.execute("TaskTracker", {"action": "create", "content": "Task B"})

        result = await handler.execute("TaskTracker", {"action": "list"})
        assert result["status"] == "success"
        assert len(result["tasks"]) == 2

    async def test_unknown_action(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        result = await handler.execute("TaskTracker", {"action": "delete"})
        assert result["status"] == "error"


class TestCreatePlanTool:
    async def test_create_plan(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        result = await handler.execute(
            "CreatePlan",
            {
                "goal": "Add authentication",
                "steps": ["Create schema", "Implement endpoints", "Write tests"],
            },
        )
        assert result["status"] == "success"
        assert wm.plan is not None
        assert wm.plan.goal == "Add authentication"
        assert len(wm.plan.steps) == 3

    async def test_create_plan_missing_goal(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        result = await handler.execute("CreatePlan", {"steps": ["step 1"]})
        assert result["status"] == "error"

    async def test_create_plan_empty_steps(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        result = await handler.execute("CreatePlan", {"goal": "Do something", "steps": []})
        assert result["status"] == "error"

    async def test_create_plan_replaces_existing(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        await handler.execute("CreatePlan", {"goal": "Old plan", "steps": ["old step"]})
        await handler.execute("CreatePlan", {"goal": "New plan", "steps": ["new step"]})
        assert wm.plan is not None
        assert wm.plan.goal == "New plan"
