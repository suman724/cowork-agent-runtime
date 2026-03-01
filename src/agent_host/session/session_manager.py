"""Session Manager — central lifecycle coordinator for the agent host."""

from __future__ import annotations

import asyncio
import contextlib
import platform
from typing import TYPE_CHECKING, Any

import structlog
from cowork_platform.policy_bundle import PolicyBundle
from cowork_platform.session_cancel_request import SessionCancelRequest
from cowork_platform.session_create_request import (
    ClientInfo,
    SessionCreateRequest,
    WorkspaceHint,
)
from cowork_platform.session_create_response import SessionCreateResponse  # noqa: TC002
from cowork_platform_sdk import CapabilityName
from google.adk.runners import Runner
from google.genai import types

from agent_host.agent.agent_factory import create_agent
from agent_host.agent.tool_adapter import TOOL_CAPABILITY_MAP
from agent_host.exceptions import (
    NoActiveTaskError,
    SessionNotFoundError,
)
from agent_host.models import SessionContext
from agent_host.session.checkpoint_session_service import CheckpointSessionService
from agent_host.session.session_client import SessionClient
from agent_host.session.workspace_client import WorkspaceClient

if TYPE_CHECKING:
    from agent_host.budget.token_budget import TokenBudget
    from agent_host.config import AgentHostConfig
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.server.stdio_transport import StdioTransport
    from tool_runtime import ToolRouter

logger = structlog.get_logger()

APP_NAME = "cowork"
VERSION = "0.1.0"


class SessionManager:
    """Central lifecycle coordinator for the agent host.

    Manages: session creation → agent initialization → task execution → cleanup.
    """

    def __init__(
        self,
        config: AgentHostConfig,
        tool_router: ToolRouter,
        transport: StdioTransport | None = None,
    ) -> None:
        self._config = config
        self._tool_router = tool_router
        self._transport = transport

        # Service clients
        self._session_client = SessionClient(config.session_service_url)
        self._workspace_client = WorkspaceClient(config.workspace_service_url)

        # ADK components (initialized on create_session)
        self._checkpoint_service = CheckpointSessionService(config.checkpoint_dir)
        self._runner: Runner | None = None
        self._token_budget: TokenBudget | None = None

        # Event emitter (created lazily after session context is available)
        self._event_emitter: EventEmitter | None = None

        # Session state
        self._session_context: SessionContext | None = None
        self._session_response: SessionCreateResponse | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._current_task_id: str | None = None

    @property
    def session_context(self) -> SessionContext | None:
        """Return the current session context, if any."""
        return self._session_context

    async def create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a new session by handshaking with the Session Service.

        Args:
            params: JSON-RPC params with tenantId, userId, workspaceHint, etc.

        Returns:
            Dict with sessionId, workspaceId, status.
        """
        tenant_id = params.get("tenantId", "")
        user_id = params.get("userId", "")

        # Build workspace hint
        workspace_hint = None
        hint_data = params.get("workspaceHint")
        if hint_data and isinstance(hint_data, dict):
            try:
                workspace_hint = WorkspaceHint.model_validate(hint_data)
            except Exception:
                logger.warning("invalid_workspace_hint", hint_data=hint_data, exc_info=True)

        # Derive capabilities from actual tools available in the router
        available_tools = self._tool_router.get_available_tools()
        capability_set: set[str] = set()
        for tool_def in available_tools:
            cap = TOOL_CAPABILITY_MAP.get(tool_def.toolName)
            if cap:
                capability_set.add(cap)
        # LLM.Call is always needed for the agent loop
        capability_set.add(CapabilityName.LLM_CALL)
        capabilities = sorted(capability_set)

        request = SessionCreateRequest(
            tenantId=tenant_id,
            userId=user_id,
            executionEnvironment="desktop",
            workspaceHint=workspace_hint,
            clientInfo=ClientInfo(
                desktopAppVersion=VERSION,
                localAgentHostVersion=VERSION,
                osFamily=platform.system(),
                osVersion=platform.release(),
            ),
            supportedCapabilities=capabilities,
        )

        # Call Session Service
        response = await self._session_client.create_session(request)
        self._session_response = response

        if response.compatibilityStatus != "compatible":
            return {
                "sessionId": response.sessionId,
                "status": "incompatible",
                "error": "Client version incompatible with server",
            }

        # Store session context
        self._session_context = SessionContext(
            session_id=response.sessionId,
            workspace_id=response.workspaceId,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        # Create EventEmitter now that session context is available
        from agent_host.events.event_emitter import EventEmitter

        self._event_emitter = EventEmitter(self._session_context, self._transport)

        # Initialize ADK agent with policy bundle
        if response.policyBundle:
            # Cast to canonical PolicyBundle type — both classes are structurally
            # identical but codegen produces separate types in each response model.
            canonical_bundle = PolicyBundle.model_validate(response.policyBundle.model_dump())
            agent, token_budget = create_agent(
                config=self._config,
                policy_bundle=canonical_bundle,
                tool_router=self._tool_router,
                event_emitter=self._event_emitter,
                workspace_client=self._workspace_client,
                session_id=response.sessionId,
            )
            self._token_budget = token_budget

            # Create ADK Runner
            self._runner = Runner(
                agent=agent,
                app_name=APP_NAME,
                session_service=self._checkpoint_service,
            )

            # Create ADK session
            await self._checkpoint_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=response.sessionId,
                state={
                    "workspace_id": response.workspaceId,
                    "tenant_id": tenant_id,
                },
            )

        # Emit session_created event
        if self._event_emitter:
            self._event_emitter.emit_session_created()

        logger.info(
            "session_created",
            session_id=response.sessionId,
            workspace_id=response.workspaceId,
        )

        return {
            "sessionId": response.sessionId,
            "workspaceId": response.workspaceId,
            "logDir": self._config.log_dir,
            "status": "ready",
        }

    async def start_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """Start a new task (agent work cycle) from a user prompt.

        Spawns the ADK Runner as an asyncio task and returns immediately.
        """
        if not self._session_context or not self._runner:
            raise SessionNotFoundError("No active session")

        task_id = params.get("taskId", "")
        prompt = params.get("prompt", "")

        # Spawn the runner as a background asyncio task.
        # Assign both fields together so cancel_task() never sees a partial state.
        task = asyncio.create_task(self._run_agent(prompt, task_id))
        self._current_task = task
        self._current_task_id = task_id

        return {"taskId": task_id, "status": "running"}

    async def _run_agent(self, prompt: str, task_id: str) -> None:
        """Run the ADK agent loop for a single task."""
        if not self._runner or not self._session_context:
            return

        try:
            message = types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            )

            assistant_text_parts: list[str] = []

            async for event in self._runner.run_async(
                user_id=self._session_context.user_id,
                session_id=self._session_context.session_id,
                new_message=message,
            ):
                # Forward streaming events
                if self._event_emitter and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            self._event_emitter.emit_text_chunk(task_id, part.text)
                            assistant_text_parts.append(part.text)

                # Record token usage from LLM responses
                if self._token_budget and hasattr(event, "usage_metadata") and event.usage_metadata:
                    usage = event.usage_metadata
                    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                    if input_tokens or output_tokens:
                        self._token_budget.record_usage(input_tokens, output_tokens)

            # Upload session history (best-effort)
            await self._upload_history(prompt, "".join(assistant_text_parts), task_id)

            # Emit task_completed after the loop finishes normally
            if self._event_emitter:
                self._event_emitter.emit_task_completed(task_id)

            logger.info("task_completed", task_id=task_id)

        except asyncio.CancelledError:
            logger.info("task_cancelled", task_id=task_id)
        except Exception as exc:
            logger.exception("task_failed", task_id=task_id)
            if self._event_emitter:
                self._event_emitter.emit_task_failed(task_id, reason=str(exc))

    async def _upload_history(self, prompt: str, assistant_text: str, task_id: str) -> None:
        """Upload session history to workspace service (best-effort)."""
        if not self._session_context:
            return

        try:
            from datetime import UTC, datetime

            from cowork_platform.conversation_message import ConversationMessage

            now = datetime.now(tz=UTC)
            messages = [
                ConversationMessage(
                    messageId=f"{task_id}-user",
                    sessionId=self._session_context.session_id,
                    taskId=task_id,
                    role="user",
                    content=prompt,
                    timestamp=now,
                ),
                ConversationMessage(
                    messageId=f"{task_id}-assistant",
                    sessionId=self._session_context.session_id,
                    taskId=task_id,
                    role="assistant",
                    content=assistant_text,
                    timestamp=now,
                ),
            ]

            await self._workspace_client.upload_session_history(
                workspace_id=self._session_context.workspace_id,
                session_id=self._session_context.session_id,
                messages=messages,
                task_id=task_id,
            )
            logger.info("session_history_uploaded", task_id=task_id)
        except Exception:
            logger.warning("session_history_upload_failed", task_id=task_id, exc_info=True)

    async def cancel_task(self) -> dict[str, Any]:
        """Cancel the currently running task (cooperative)."""
        if not self._current_task:
            raise NoActiveTaskError("No task is currently running")

        self._current_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._current_task

        task_id = self._current_task_id or ""
        self._current_task = None
        self._current_task_id = None

        return {"taskId": task_id, "status": "cancelled"}

    async def deliver_approval(self, params: dict[str, Any]) -> dict[str, Any]:
        """Deliver a user approval/denial decision for a pending tool call."""
        approval_id = params.get("approvalId", "")
        decision = params.get("decision", "denied")

        logger.info(
            "approval_delivered",
            approval_id=approval_id,
            decision=decision,
        )

        # In Phase 1, approval is handled via LongRunningFunctionTool
        # The Desktop App sends the updated function response back
        return {
            "approvalId": approval_id,
            "status": "delivered",
            "decision": decision,
        }

    async def get_session_state(self) -> dict[str, Any]:
        """Return current session and task status."""
        if not self._session_context:
            return {"status": "no_session"}

        state: dict[str, Any] = {
            "sessionId": self._session_context.session_id,
            "workspaceId": self._session_context.workspace_id,
            "hasActiveTask": self._current_task is not None and not self._current_task.done(),
            "currentTaskId": self._current_task_id,
        }

        if self._token_budget:
            state["tokenUsage"] = {
                "inputTokens": self._token_budget.input_tokens_used,
                "outputTokens": self._token_budget.output_tokens_used,
                "totalTokens": self._token_budget.total_tokens_used,
                "remaining": self._token_budget.remaining,
                "maxSessionTokens": self._token_budget.max_session_tokens,
            }

        return state

    async def shutdown(self) -> dict[str, Any]:
        """Clean session teardown."""
        # Cancel any running task
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._current_task

        session_id = ""
        if self._session_context:
            session_id = self._session_context.session_id

            # Cancel session on backend (best-effort)
            try:
                await self._session_client.cancel_session(
                    session_id,
                    SessionCancelRequest(reason="shutdown"),
                )
            except Exception:
                logger.warning("shutdown_cancel_failed", exc_info=True)

            # Delete checkpoint (clean exit)
            if self._session_context:
                await self._checkpoint_service.delete_session(
                    app_name=APP_NAME,
                    user_id=self._session_context.user_id,
                    session_id=session_id,
                )

        # Emit session_completed before cleanup
        if self._event_emitter:
            self._event_emitter.emit_session_completed()

        # Close HTTP clients
        await self._session_client.close()
        await self._workspace_client.close()

        self._session_context = None
        self._runner = None
        self._current_task = None

        logger.info("session_shutdown", session_id=session_id)
        return {"status": "shutdown", "sessionId": session_id}
