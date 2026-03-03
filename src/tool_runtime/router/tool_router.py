"""ToolRouter — registry, dispatch, error mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from cowork_platform.tool_definition import ToolDefinition
from cowork_platform.tool_result import Error, ToolResult

from tool_runtime.exceptions import ToolNotFoundError, ToolRuntimeError
from tool_runtime.models import ExecutionContext, ToolExecutionResult
from tool_runtime.platform.detection import get_platform
from tool_runtime.tools.code.execute_code import ExecuteCodeTool
from tool_runtime.tools.file.create_directory import CreateDirectoryTool
from tool_runtime.tools.file.delete_file import DeleteFileTool
from tool_runtime.tools.file.edit_file import EditFileTool
from tool_runtime.tools.file.find_files import FindFilesTool
from tool_runtime.tools.file.grep_files import GrepFilesTool
from tool_runtime.tools.file.list_directory import ListDirectoryTool
from tool_runtime.tools.file.move_file import MoveFileTool
from tool_runtime.tools.file.multi_edit import MultiEditTool
from tool_runtime.tools.file.read_file import ReadFileTool
from tool_runtime.tools.file.view_image import ViewImageTool
from tool_runtime.tools.file.write_file import WriteFileTool
from tool_runtime.tools.network.fetch_url import FetchUrlTool
from tool_runtime.tools.network.http_request import HttpRequestTool
from tool_runtime.tools.network.web_search import WebSearchTool
from tool_runtime.tools.shell.run_command import RunCommandTool

if TYPE_CHECKING:
    import httpx
    from cowork_platform.tool_request import ToolRequest

    from tool_runtime.platform.base import PlatformAdapter
    from tool_runtime.tools.base import BaseTool

logger = structlog.get_logger(__name__)


class ToolRouter:
    """Routes tool requests to registered tool implementations.

    Never raises — all errors are captured in the returned ToolExecutionResult.
    """

    def __init__(
        self,
        platform: PlatformAdapter | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._platform = platform or get_platform()
        self._tools: dict[str, BaseTool] = {}

        # Register built-in tools
        self._register(ReadFileTool(self._platform))
        self._register(WriteFileTool(self._platform))
        self._register(DeleteFileTool(self._platform))
        self._register(EditFileTool(self._platform))
        self._register(MultiEditTool(self._platform))
        self._register(ListDirectoryTool(self._platform))
        self._register(CreateDirectoryTool(self._platform))
        self._register(MoveFileTool(self._platform))
        self._register(ViewImageTool(self._platform))
        self._register(FindFilesTool())
        self._register(GrepFilesTool())
        self._register(RunCommandTool(self._platform))
        self._register(ExecuteCodeTool(self._platform))
        self._register(HttpRequestTool(http_client))
        self._register(FetchUrlTool(http_client))
        self._register(WebSearchTool(http_client))

    def _register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    async def execute(
        self,
        request: ToolRequest,
        context: ExecutionContext | None = None,
    ) -> ToolExecutionResult:
        """Dispatch a tool request and return the result.

        Never raises — all errors are captured in the result with
        status="failed" and an appropriate error code.
        """
        ctx = context or ExecutionContext()

        try:
            tool = self._tools.get(request.toolName)
            if tool is None:
                raise ToolNotFoundError(f"Unknown tool: {request.toolName}")

            # Each tool's execute() calls validate_input internally
            raw_output = await tool.execute(request.arguments, ctx)

            artifacts = []
            artifact_uris: list[str] = []
            if raw_output.artifact_data is not None:
                artifacts.append(raw_output.artifact_data)
                # URI will be populated by agent_host after upload
                artifact_uris.append(f"pending://{raw_output.artifact_data.artifact_name}")

            tool_result = ToolResult(
                toolName=request.toolName,
                sessionId=request.sessionId,
                taskId=request.taskId,
                stepId=request.stepId,
                status="succeeded",
                outputText=raw_output.output_text,
                artifactUris=artifact_uris if artifact_uris else None,
            )

            return ToolExecutionResult(
                tool_result=tool_result,
                artifacts=artifacts,
                image_content=raw_output.image_content,
            )

        except ToolRuntimeError as e:
            logger.warning(
                "tool_execution_failed",
                tool_name=request.toolName,
                error_code=e.code,
                error_message=e.message,
            )
            tool_result = ToolResult(
                toolName=request.toolName,
                sessionId=request.sessionId,
                taskId=request.taskId,
                stepId=request.stepId,
                status="failed",
                error=Error(code=e.code, message=e.message),
            )
            return ToolExecutionResult(tool_result=tool_result)

        except Exception as e:
            logger.exception(
                "unexpected_tool_error",
                tool_name=request.toolName,
            )
            tool_result = ToolResult(
                toolName=request.toolName,
                sessionId=request.sessionId,
                taskId=request.taskId,
                stepId=request.stepId,
                status="failed",
                error=Error(
                    code="TOOL_EXECUTION_FAILED",
                    message=f"Unexpected error: {e}",
                ),
            )
            return ToolExecutionResult(tool_result=tool_result)

    def get_available_tools(self) -> list[ToolDefinition]:
        """Return definitions for all registered tools."""
        return [
            ToolDefinition(
                toolName=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            )
            for tool in self._tools.values()
        ]
