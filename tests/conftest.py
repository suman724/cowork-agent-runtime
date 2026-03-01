"""Shared fixtures for tool runtime tests."""

from __future__ import annotations

from typing import Any

import pytest
from cowork_platform.tool_request import ToolRequest

from tool_runtime.models import ExecutionContext
from tool_runtime.platform.darwin import DarwinAdapter
from tool_runtime.router.tool_router import ToolRouter


@pytest.fixture
def platform_adapter() -> DarwinAdapter:
    """Provide a Darwin platform adapter for tests."""
    return DarwinAdapter()


@pytest.fixture
def tool_router(platform_adapter: DarwinAdapter) -> ToolRouter:
    """Provide a ToolRouter with default tools registered."""
    return ToolRouter(platform=platform_adapter)


@pytest.fixture
def execution_context(tmp_path: object) -> ExecutionContext:
    """Provide a default execution context with tmp_path as working directory."""
    return ExecutionContext(working_directory=str(tmp_path))


def make_tool_request(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    session_id: str = "sess-test",
    task_id: str = "task-test",
    step_id: str = "step-test",
) -> ToolRequest:
    """Helper to create a ToolRequest for testing."""
    return ToolRequest(
        toolName=tool_name,
        arguments=arguments or {},
        sessionId=session_id,
        taskId=task_id,
        stepId=step_id,
    )
