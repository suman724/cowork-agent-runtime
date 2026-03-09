"""Tests for team tool definitions."""

from __future__ import annotations

from agent_host.teams.tools import (
    ALL_TEAM_TOOL_NAMES,
    LEAD_TOOL_NAMES,
    LEAD_TOOLS,
    SHARED_TOOL_NAMES,
    SHARED_TOOLS,
)

EXPECTED_LEAD = {"CreateTeam", "CreateTeammate", "ShutdownTeammate", "ShutdownTeam", "WaitForTeam"}
EXPECTED_SHARED = {"TeamTaskCreate", "TeamTaskUpdate", "TeamTaskList", "SendTeamMessage"}


class TestToolDefinitions:
    def test_lead_tools_count(self) -> None:
        assert len(LEAD_TOOLS) == 5

    def test_shared_tools_count(self) -> None:
        assert len(SHARED_TOOLS) == 4

    def test_all_tools_have_function_type(self) -> None:
        for tool in LEAD_TOOLS + SHARED_TOOLS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]

    def test_lead_tool_names(self) -> None:
        assert LEAD_TOOL_NAMES == EXPECTED_LEAD

    def test_shared_tool_names(self) -> None:
        assert SHARED_TOOL_NAMES == EXPECTED_SHARED

    def test_all_tool_names_is_union(self) -> None:
        assert ALL_TEAM_TOOL_NAMES == LEAD_TOOL_NAMES | SHARED_TOOL_NAMES

    def test_no_overlap(self) -> None:
        assert not (LEAD_TOOL_NAMES & SHARED_TOOL_NAMES)

    def test_required_fields_present(self) -> None:
        for tool in LEAD_TOOLS + SHARED_TOOLS:
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
