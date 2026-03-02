"""Session Manager — central lifecycle coordinator for the agent host."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import platform
import random
from typing import TYPE_CHECKING, Any

import structlog
from cowork_platform.conversation_message import ConversationMessage
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
    CheckpointError,
    NoActiveTaskError,
    PolicyExpiredError,
    SessionNotFoundError,
)
from agent_host.llm.error_classifier import is_transient_llm_error
from agent_host.models import SessionContext
from agent_host.session.checkpoint_session_service import CheckpointSessionService
from agent_host.session.session_client import SessionClient
from agent_host.session.workspace_client import WorkspaceClient

if TYPE_CHECKING:
    from agent_host.agent.file_change_tracker import FileChangeTracker
    from agent_host.approval.approval_gate import ApprovalGate
    from agent_host.budget.token_budget import TokenBudget
    from agent_host.config import AgentHostConfig
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.server.stdio_transport import StdioTransport
    from tool_runtime import ToolRouter

logger = structlog.get_logger()

APP_NAME = "cowork"
VERSION = "0.1.0"

# Maximum length for tool output stored in history messages
_MAX_TOOL_OUTPUT_LENGTH = 4000


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

        # Approval and file tracking (set after create_agent)
        self._approval_gate: ApprovalGate | None = None
        self._file_change_tracker: FileChangeTracker | None = None

        # Session state
        self._session_context: SessionContext | None = None
        self._session_response: SessionCreateResponse | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._current_task_id: str | None = None

        # Rich history: tool messages collected during a task
        self._task_tool_messages: list[dict[str, str]] = []

        # Cumulative session history (across tasks)
        self._session_messages: list[ConversationMessage] = []

        # Step tracking for current task
        self._current_max_steps: int = config.default_max_steps
        self._current_step_count: int = 0

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
            components = create_agent(
                config=self._config,
                policy_bundle=canonical_bundle,
                tool_router=self._tool_router,
                event_emitter=self._event_emitter,
                workspace_client=self._workspace_client,
                session_id=response.sessionId,
                workspace_id=response.workspaceId,
            )
            self._token_budget = components.token_budget
            self._approval_gate = components.approval_gate
            self._file_change_tracker = components.file_change_tracker

            # Create ADK Runner
            self._runner = Runner(
                agent=components.agent,
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

        # Reset cumulative history for the new session
        self._session_messages = []

        # Restore token budget and session messages from checkpoint (crash recovery)
        self._restore_token_budget_from_checkpoint()
        self._restore_session_messages_from_checkpoint()

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

    async def resume_session(self, params: dict[str, Any]) -> dict[str, Any]:
        """Resume an existing session — reuses the same session ID.

        Calls the Session Service's resume endpoint, re-initializes the ADK agent,
        and restores cumulative session history from the Workspace Service.
        """
        session_id = params.get("sessionId", "")
        if not session_id:
            raise SessionNotFoundError("sessionId is required for ResumeSession")

        # Resume via Session Service (refreshes policy, extends expiry)
        response = await self._session_client.resume_session(session_id)

        # Store session context
        self._session_context = SessionContext(
            session_id=response.sessionId,
            workspace_id=response.workspaceId,
            tenant_id="",  # Not returned by resume; not needed for agent loop
            user_id="",
        )

        # Create EventEmitter
        from agent_host.events.event_emitter import EventEmitter

        self._event_emitter = EventEmitter(self._session_context, self._transport)

        # Initialize ADK agent with refreshed policy bundle
        if response.policyBundle:
            canonical_bundle = PolicyBundle.model_validate(response.policyBundle.model_dump())
            components = create_agent(
                config=self._config,
                policy_bundle=canonical_bundle,
                tool_router=self._tool_router,
                event_emitter=self._event_emitter,
                workspace_client=self._workspace_client,
                session_id=response.sessionId,
                workspace_id=response.workspaceId,
            )
            self._token_budget = components.token_budget
            self._approval_gate = components.approval_gate
            self._file_change_tracker = components.file_change_tracker

            self._runner = Runner(
                agent=components.agent,
                app_name=APP_NAME,
                session_service=self._checkpoint_service,
            )

            await self._checkpoint_service.create_session(
                app_name=APP_NAME,
                user_id="",
                session_id=response.sessionId,
                state={
                    "workspace_id": response.workspaceId,
                },
            )

        # Restore cumulative history from Workspace Service
        self._session_messages = []
        prior_messages = await self._workspace_client.get_session_history(
            workspace_id=response.workspaceId,
            session_id=response.sessionId,
        )
        if prior_messages:
            self._session_messages = prior_messages
            logger.info(
                "session_history_restored",
                session_id=response.sessionId,
                message_count=len(prior_messages),
            )

        # Restore token budget from checkpoint (if available)
        self._restore_token_budget_from_checkpoint()

        if self._event_emitter:
            self._event_emitter.emit_session_created()

        logger.info(
            "session_resumed",
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

        # Parse taskOptions - extract maxSteps (clamped 1-200)
        task_options = params.get("taskOptions") or {}
        raw_max_steps = task_options.get("maxSteps", self._config.default_max_steps)
        try:
            max_steps = int(raw_max_steps)
        except (TypeError, ValueError):
            max_steps = self._config.default_max_steps
        max_steps = max(1, min(200, max_steps))
        self._current_max_steps = max_steps
        self._current_step_count = 0

        # Spawn the runner as a background asyncio task.
        # Assign both fields together so cancel_task() never sees a partial state.
        task = asyncio.create_task(self._run_agent(prompt, task_id, max_steps))
        self._current_task = task
        self._current_task_id = task_id

        return {"taskId": task_id, "status": "running"}

    async def _run_agent(self, prompt: str, task_id: str, max_steps: int = 50) -> None:
        """Run the ADK agent loop for a single task with retry on transient errors."""
        if not self._runner or not self._session_context:
            return

        # Clear tool messages for this task
        self._task_tool_messages = []

        max_retries = self._config.llm_max_retries
        base_delay = self._config.llm_retry_base_delay
        max_delay = self._config.llm_retry_max_delay

        message = types.Content(
            role="user",
            parts=[types.Part(text=prompt)],
        )

        for attempt in range(max_retries + 1):
            try:
                assistant_text_parts: list[str] = []

                async for event in self._runner.run_async(
                    user_id=self._session_context.user_id,
                    session_id=self._session_context.session_id,
                    new_message=message if attempt == 0 else None,
                ):
                    # Forward streaming events
                    if self._event_emitter and event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                self._event_emitter.emit_text_chunk(task_id, part.text)
                                assistant_text_parts.append(part.text)

                            # Track tool calls for rich history
                            if hasattr(part, "function_call") and part.function_call:
                                fc = part.function_call
                                self._task_tool_messages.append(
                                    {
                                        "type": "tool_call",
                                        "toolName": getattr(fc, "name", ""),
                                        "arguments": json.dumps(
                                            getattr(fc, "args", {}), default=str
                                        ),
                                    }
                                )

                            # Track tool results for rich history
                            if hasattr(part, "function_response") and part.function_response:
                                fr = part.function_response
                                output = json.dumps(getattr(fr, "response", {}), default=str)
                                if len(output) > _MAX_TOOL_OUTPUT_LENGTH:
                                    output = output[:_MAX_TOOL_OUTPUT_LENGTH] + "... [truncated]"
                                self._task_tool_messages.append(
                                    {
                                        "type": "tool_result",
                                        "toolName": getattr(fr, "name", ""),
                                        "output": output,
                                    }
                                )

                    # Record token usage and count steps from LLM responses
                    if hasattr(event, "usage_metadata") and event.usage_metadata:
                        if self._token_budget:
                            usage = event.usage_metadata
                            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                            if input_tokens or output_tokens:
                                self._token_budget.record_usage(input_tokens, output_tokens)
                                self._persist_token_budget()

                        # Step counting: each usage_metadata = one LLM call = one step
                        self._current_step_count += 1
                        step_count = self._current_step_count
                        warning_threshold = math.floor(max_steps * 0.8)
                        if step_count == warning_threshold and self._event_emitter:
                            self._event_emitter.emit_step_limit_approaching(
                                task_id, step_count, max_steps
                            )
                        if step_count >= max_steps:
                            if self._event_emitter:
                                self._event_emitter.emit_task_failed(
                                    task_id,
                                    reason=f"Step limit reached ({step_count}/{max_steps})",
                                )
                            logger.warning(
                                "step_limit_reached",
                                task_id=task_id,
                                step_count=step_count,
                                max_steps=max_steps,
                            )
                            return

                # Upload session history (best-effort)
                await self._upload_history(prompt, "".join(assistant_text_parts), task_id)

                # Emit task_completed after the loop finishes normally
                if self._event_emitter:
                    self._event_emitter.emit_task_completed(task_id)

                logger.info("task_completed", task_id=task_id)
                return

            except asyncio.CancelledError:
                logger.info("task_cancelled", task_id=task_id)
                return

            except Exception as exc:
                if is_transient_llm_error(exc) and attempt < max_retries:
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.25)  # noqa: S311
                    delay += jitter
                    logger.warning(
                        "llm_transient_error_retrying",
                        task_id=task_id,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=delay,
                        error=str(exc),
                    )
                    if self._event_emitter:
                        self._event_emitter.emit_llm_retry(
                            task_id=task_id,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            error_message=str(exc),
                            delay_seconds=delay,
                        )
                    await asyncio.sleep(delay)
                    continue

                # Unrecoverable — classify as session-level or task-level failure
                if isinstance(exc, (PolicyExpiredError, CheckpointError)):
                    logger.exception("session_failed", task_id=task_id)
                    if self._event_emitter:
                        self._event_emitter.emit_session_failed(str(exc))
                else:
                    logger.exception("task_failed", task_id=task_id)
                    if self._event_emitter:
                        self._event_emitter.emit_task_failed(task_id, reason=str(exc))
                return

    def _persist_token_budget(self) -> None:
        """Persist current token usage into the ADK session checkpoint."""
        if not self._token_budget or not self._session_context:
            return

        session_id = self._session_context.session_id
        session = self._checkpoint_service._sessions.get(session_id)
        if not session:
            return

        session.state["_token_input_used"] = self._token_budget.input_tokens_used
        session.state["_token_output_used"] = self._token_budget.output_tokens_used
        self._checkpoint_service._write_checkpoint(APP_NAME, session)

    def _restore_token_budget_from_checkpoint(self) -> None:
        """Restore token usage from checkpoint state if available."""
        if not self._token_budget or not self._session_context:
            return

        session_id = self._session_context.session_id
        session = self._checkpoint_service._sessions.get(session_id)
        if not session:
            return

        input_used = session.state.get("_token_input_used")
        output_used = session.state.get("_token_output_used")
        if isinstance(input_used, int) and isinstance(output_used, int):
            self._token_budget.restore_usage(input_used, output_used)
            logger.info(
                "token_budget_restored",
                input_tokens=input_used,
                output_tokens=output_used,
            )

    def _persist_session_messages(self) -> None:
        """Persist cumulative session messages into the ADK session checkpoint."""
        if not self._session_context:
            return

        session_id = self._session_context.session_id
        session = self._checkpoint_service._sessions.get(session_id)
        if not session:
            return

        session.state["_session_messages"] = [
            msg.model_dump(mode="json") for msg in self._session_messages
        ]
        self._checkpoint_service._write_checkpoint(APP_NAME, session)

    def _restore_session_messages_from_checkpoint(self) -> None:
        """Restore cumulative session messages from checkpoint state if available."""
        if not self._session_context:
            return

        session_id = self._session_context.session_id
        session = self._checkpoint_service._sessions.get(session_id)
        if not session:
            return

        raw = session.state.get("_session_messages")
        if isinstance(raw, list):
            try:
                self._session_messages = [ConversationMessage.model_validate(msg) for msg in raw]
                logger.info(
                    "session_messages_restored",
                    count=len(self._session_messages),
                )
            except Exception:
                logger.warning("session_messages_restore_failed", exc_info=True)

    async def _upload_history(self, prompt: str, assistant_text: str, task_id: str) -> None:
        """Upload cumulative session history to workspace service (best-effort).

        Appends this task's messages to _session_messages, then uploads the full list.
        Caps at 500 messages (drops oldest, keeps first system message).
        """
        if not self._session_context:
            return

        try:
            from datetime import UTC, datetime

            now = datetime.now(tz=UTC)
            task_messages: list[ConversationMessage] = [
                ConversationMessage(
                    messageId=f"{task_id}-user",
                    sessionId=self._session_context.session_id,
                    taskId=task_id,
                    role="user",
                    content=prompt,
                    timestamp=now,
                ),
            ]

            # Insert tool call/result messages
            for i, tool_msg in enumerate(self._task_tool_messages):
                task_messages.append(
                    ConversationMessage(
                        messageId=f"{task_id}-tool-{i}",
                        sessionId=self._session_context.session_id,
                        taskId=task_id,
                        role="tool",
                        content=json.dumps(tool_msg, default=str),
                        timestamp=now,
                    )
                )

            task_messages.append(
                ConversationMessage(
                    messageId=f"{task_id}-assistant",
                    sessionId=self._session_context.session_id,
                    taskId=task_id,
                    role="assistant",
                    content=assistant_text,
                    timestamp=now,
                ),
            )

            # Accumulate into session-level history
            self._session_messages.extend(task_messages)

            # Cap at 500 messages (drop oldest, keep first)
            max_history = 500
            if len(self._session_messages) > max_history:
                self._session_messages = (
                    self._session_messages[:1] + self._session_messages[-(max_history - 1) :]
                )

            await self._workspace_client.upload_session_history(
                workspace_id=self._session_context.workspace_id,
                session_id=self._session_context.session_id,
                messages=self._session_messages,
                task_id=task_id,
            )

            # Persist to checkpoint for crash recovery
            self._persist_session_messages()

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

        delivered = False
        if self._approval_gate:
            delivered = self._approval_gate.deliver(approval_id, decision)

        if not delivered:
            logger.warning(
                "approval_not_pending",
                approval_id=approval_id,
            )

        return {
            "approvalId": approval_id,
            "status": "delivered" if delivered else "not_found",
            "decision": decision,
        }

    async def get_patch_preview(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a patch preview (unified diffs) for a task's file changes."""
        task_id = params.get("taskId", self._current_task_id or "")

        if not self._file_change_tracker:
            return {"taskId": task_id, "files": []}

        return self._file_change_tracker.get_patch_preview(task_id)

    async def get_session_state(self) -> dict[str, Any]:
        """Return current session and task status."""
        if not self._session_context:
            return {"status": "no_session"}

        state: dict[str, Any] = {
            "sessionId": self._session_context.session_id,
            "workspaceId": self._session_context.workspace_id,
            "hasActiveTask": self._current_task is not None and not self._current_task.done(),
            "currentTaskId": self._current_task_id,
            "currentStep": self._current_step_count,
            "maxSteps": self._current_max_steps,
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
        # Check for active task *before* cancelling it
        had_active_task = bool(self._current_task and not self._current_task.done())

        # Cancel any running task
        if had_active_task and self._current_task:
            self._current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._current_task

        session_id = ""
        if self._session_context:
            session_id = self._session_context.session_id

            # Only cancel session on backend if there was a running task at shutdown.
            # Completed/failed sessions are left as-is so they can be resumed later.
            if had_active_task:
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
        self._approval_gate = None
        self._file_change_tracker = None

        logger.info("session_shutdown", session_id=session_id)
        return {"status": "shutdown", "sessionId": session_id}
