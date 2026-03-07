"""Tests for AgentToolHandler — internal tools for working memory."""

from __future__ import annotations

from typing import Any

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

    async def test_create_plan_fires_on_plan_updated(self) -> None:
        wm = WorkingMemory()
        calls: list[tuple[str, list[dict[str, Any]]]] = []
        handler = AgentToolHandler(
            wm,
            on_plan_updated=lambda goal, steps: calls.append((goal, steps)),
        )
        await handler.execute(
            "CreatePlan", {"goal": "Build feature", "steps": ["Step A", "Step B"]}
        )
        assert len(calls) == 1
        assert calls[0][0] == "Build feature"
        assert len(calls[0][1]) == 2
        assert calls[0][1][0]["status"] == "pending"


class TestUpdatePlanStepTool:
    async def test_update_step_to_in_progress(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        await handler.execute("CreatePlan", {"goal": "Goal", "steps": ["A", "B", "C"]})

        result = await handler.execute("UpdatePlanStep", {"stepIndex": 0, "status": "in_progress"})
        assert result["status"] == "success"
        assert result["stepIndex"] == 0
        assert result["newStatus"] == "in_progress"
        assert result["description"] == "A"
        assert wm.plan is not None
        assert wm.plan.steps[0].status == "in_progress"

    async def test_update_step_to_completed(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        await handler.execute("CreatePlan", {"goal": "Goal", "steps": ["A", "B"]})

        result = await handler.execute("UpdatePlanStep", {"stepIndex": 1, "status": "completed"})
        assert result["status"] == "success"
        assert wm.plan is not None
        assert wm.plan.steps[1].status == "completed"

    async def test_update_step_to_skipped(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        await handler.execute("CreatePlan", {"goal": "Goal", "steps": ["A"]})

        result = await handler.execute("UpdatePlanStep", {"stepIndex": 0, "status": "skipped"})
        assert result["status"] == "success"
        assert wm.plan is not None
        assert wm.plan.steps[0].status == "skipped"

    async def test_no_plan_exists(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        result = await handler.execute("UpdatePlanStep", {"stepIndex": 0, "status": "completed"})
        assert result["status"] == "error"
        assert "No plan exists" in result["message"]

    async def test_missing_step_index(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        await handler.execute("CreatePlan", {"goal": "Goal", "steps": ["A"]})

        result = await handler.execute("UpdatePlanStep", {"status": "completed"})
        assert result["status"] == "error"
        assert "stepIndex" in result["message"]

    async def test_invalid_step_index_type(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        await handler.execute("CreatePlan", {"goal": "Goal", "steps": ["A"]})

        result = await handler.execute(
            "UpdatePlanStep", {"stepIndex": "zero", "status": "completed"}
        )
        assert result["status"] == "error"
        assert "stepIndex" in result["message"]

    async def test_step_index_out_of_range(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        await handler.execute("CreatePlan", {"goal": "Goal", "steps": ["A", "B"]})

        result = await handler.execute("UpdatePlanStep", {"stepIndex": 5, "status": "completed"})
        assert result["status"] == "error"
        assert "out of range" in result["message"]

    async def test_negative_step_index(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        await handler.execute("CreatePlan", {"goal": "Goal", "steps": ["A"]})

        result = await handler.execute("UpdatePlanStep", {"stepIndex": -1, "status": "completed"})
        assert result["status"] == "error"
        assert "out of range" in result["message"]

    async def test_invalid_status(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)
        await handler.execute("CreatePlan", {"goal": "Goal", "steps": ["A"]})

        result = await handler.execute("UpdatePlanStep", {"stepIndex": 0, "status": "done"})
        assert result["status"] == "error"
        assert "status must be" in result["message"]

    async def test_fires_on_plan_updated_callback(self) -> None:
        wm = WorkingMemory()
        calls: list[tuple[str, list[dict[str, Any]]]] = []
        handler = AgentToolHandler(
            wm,
            on_plan_updated=lambda goal, steps: calls.append((goal, steps)),
        )
        await handler.execute("CreatePlan", {"goal": "Goal", "steps": ["A", "B"]})
        calls.clear()  # Ignore the CreatePlan notification

        await handler.execute("UpdatePlanStep", {"stepIndex": 0, "status": "in_progress"})
        assert len(calls) == 1
        assert calls[0][0] == "Goal"
        assert calls[0][1][0]["status"] == "in_progress"
        assert calls[0][1][1]["status"] == "pending"

    async def test_no_callback_does_not_error(self) -> None:
        wm = WorkingMemory()
        handler = AgentToolHandler(wm)  # No on_plan_updated callback
        await handler.execute("CreatePlan", {"goal": "Goal", "steps": ["A"]})

        # Should succeed without errors even with no callback
        result = await handler.execute("UpdatePlanStep", {"stepIndex": 0, "status": "completed"})
        assert result["status"] == "success"

    def test_update_plan_step_in_tool_definitions(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        defs = handler.get_tool_definitions()
        names = {d["function"]["name"] for d in defs}
        assert "UpdatePlanStep" in names

        # Verify the definition shape
        ups_def = next(d for d in defs if d["function"]["name"] == "UpdatePlanStep")
        params = ups_def["function"]["parameters"]
        assert "stepIndex" in params["properties"]
        assert "status" in params["properties"]
        assert params["properties"]["status"]["enum"] == [
            "in_progress",
            "completed",
            "skipped",
        ]

    def test_is_agent_tool_includes_update_plan_step(self) -> None:
        handler = AgentToolHandler(WorkingMemory())
        assert handler.is_agent_tool("UpdatePlanStep")
