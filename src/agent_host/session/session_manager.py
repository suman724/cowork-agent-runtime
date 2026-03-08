"""Session Manager — central lifecycle coordinator for the agent host."""

from __future__ import annotations

import asyncio
import contextlib
import json
import platform
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

from agent_host.agent.file_change_tracker import FileChangeTracker
from agent_host.approval.approval_gate import ApprovalGate
from agent_host.budget.token_budget import TokenBudget
from agent_host.exceptions import (
    CheckpointError,
    NoActiveTaskError,
    PolicyExpiredError,
    SessionNotFoundError,
)
from agent_host.llm.client import LLMClient
from agent_host.loop.agent_tools import AgentToolHandler
from agent_host.loop.loop_runtime import LoopRuntime
from agent_host.loop.react_loop import ReactLoop
from agent_host.loop.system_prompt import SystemPromptBuilder
from agent_host.loop.tool_executor import TOOL_CAPABILITY_MAP, ToolExecutor
from agent_host.loop.verification import VerificationConfig
from agent_host.memory.memory_manager import MemoryManager
from agent_host.memory.working_memory import WorkingMemory
from agent_host.models import SessionContext
from agent_host.policy.policy_enforcer import PolicyEnforcer
from agent_host.session.checkpoint_manager import CheckpointManager, SessionCheckpoint
from agent_host.session.session_client import SessionClient
from agent_host.session.workspace_client import WorkspaceClient
from agent_host.skills.skill_loader import SkillLoader
from agent_host.thread.compactor import ContextCompactor, DropOldestCompactor, HybridCompactor
from agent_host.thread.message_thread import MessageThread
from tool_runtime.models import ExecutionContext

if TYPE_CHECKING:
    from agent_host.config import AgentHostConfig
    from agent_host.events.event_emitter import EventEmitter
    from agent_host.server.stdio_transport import StdioTransport
    from agent_host.skills.models import SkillDefinition
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

        # Checkpoint manager (replaces ADK CheckpointSessionService)
        self._checkpoint_manager = CheckpointManager(config.checkpoint_dir)

        # LLM client (initialized on create_session)
        self._llm_client: LLMClient | None = None

        # Event emitter (created lazily after session context is available)
        self._event_emitter: EventEmitter | None = None

        # Policy, budget, approval, file tracking (set after create_session)
        self._policy_enforcer: PolicyEnforcer | None = None
        self._token_budget: TokenBudget | None = None
        self._approval_gate: ApprovalGate | None = None
        self._file_change_tracker: FileChangeTracker | None = None

        # Message thread (replaces ADK session)
        self._thread: MessageThread | None = None

        # Working memory (task tracker, plan, notes — injected per-turn)
        self._working_memory: WorkingMemory | None = None

        # Persistent memory manager (project instructions + auto-memory)
        self._memory_manager: MemoryManager | None = None

        # Skills (loaded from built-in, workspace YAML, and policy bundle)
        self._skills: list[SkillDefinition] = []

        # Session state
        self._session_context: SessionContext | None = None
        self._session_response: SessionCreateResponse | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._current_task_id: str | None = None
        self._cancel_event: asyncio.Event = asyncio.Event()

        # Workspace directory (set from workspace hint in create_session)
        self._workspace_dir: str | None = None

        # Cumulative session history (across tasks)
        self._session_messages: list[ConversationMessage] = []

        # LLM limits (overridden by policy bundle in _init_components)
        self._max_context_tokens: int = 100_000

        # Step tracking for current task
        self._current_max_steps: int = config.default_max_steps
        self._current_step_count: int = 0

        # Active task prompt (for checkpoint persistence)
        self._current_task_prompt: str = ""

        # Last workspace sync step (for periodic sync)
        self._last_workspace_sync_step: int = 0

        # Incomplete task from crash recovery (set by _restore_from_checkpoint)
        self._incomplete_task: dict[str, Any] | None = None

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

        # Extract workspace directory from hint for system prompt context
        if workspace_hint and workspace_hint.localPaths:
            self._workspace_dir = workspace_hint.localPaths[0]

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

        # Initialize components with policy bundle
        if response.policyBundle:
            canonical_bundle = PolicyBundle.model_validate(response.policyBundle.model_dump())
            self._init_components(canonical_bundle)

        # Reset cumulative history for the new session
        self._session_messages = []

        # Restore token budget and session messages from checkpoint (crash recovery)
        await self._restore_from_checkpoint()

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

        Calls the Session Service's resume endpoint, re-initializes the agent,
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

        # Initialize components with refreshed policy bundle
        if response.policyBundle:
            canonical_bundle = PolicyBundle.model_validate(response.policyBundle.model_dump())
            self._init_components(canonical_bundle)

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
        await self._restore_from_checkpoint()

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

    def _init_components(
        self,
        policy_bundle: PolicyBundle,
    ) -> None:
        """Initialize all agent loop components from a policy bundle."""
        # Inject workspace directory into path-based capabilities so the agent
        # can operate on workspace files.  The Policy Service returns static
        # allowedPaths (e.g. "."); the workspace dir is a local concern and is
        # enriched here rather than threading local paths through the backend.
        if self._workspace_dir:
            self._inject_workspace_path(policy_bundle, self._workspace_dir)

        # Policy enforcer
        self._policy_enforcer = PolicyEnforcer(policy_bundle)

        # LLM policy limits from the policy bundle
        llm_policy = policy_bundle.llmPolicy
        max_tokens = llm_policy.maxSessionTokens if llm_policy else 100_000
        self._max_context_tokens = llm_policy.maxInputTokens if llm_policy else 100_000
        max_output_tokens = llm_policy.maxOutputTokens if llm_policy else None

        # Token budget
        self._token_budget = TokenBudget(max_session_tokens=max_tokens)

        # Approval gate and file change tracker
        self._approval_gate = ApprovalGate()
        self._file_change_tracker = FileChangeTracker()

        # LLM client
        self._llm_client = LLMClient(
            endpoint=self._config.llm_gateway_endpoint,
            auth_token=self._config.llm_gateway_auth_token,
            model=self._config.llm_model,
            max_output_tokens=max_output_tokens,
            max_retries=self._config.llm_max_retries,
            retry_base_delay=self._config.llm_retry_base_delay,
            retry_max_delay=self._config.llm_retry_max_delay,
            event_emitter=self._event_emitter,
            default_headers=self._config.llm_gateway_headers or None,
        )

        # Persistent memory manager (project instructions + auto-memory)
        if self._workspace_dir:
            self._memory_manager = MemoryManager(self._workspace_dir, config=self._config)
            self._memory_manager.load_all()

        # System prompt
        prompt_builder = SystemPromptBuilder(
            workspace_dir=self._workspace_dir,
            os_family=platform.system(),
        )

        # Message thread
        self._thread = MessageThread(
            system_prompt=prompt_builder.build_static_prompt(
                policy_enforcer=self._policy_enforcer,
                project_instructions=(
                    self._memory_manager.project_instructions if self._memory_manager else ""
                ),
                has_persistent_memory=self._memory_manager is not None,
            )
        )

        # Load skills
        skill_loader = SkillLoader()
        self._skills = skill_loader.load_all()
        if self._skills:
            logger.info("skills_loaded", count=len(self._skills))

    @staticmethod
    def _inject_workspace_path(policy_bundle: PolicyBundle, workspace_dir: str) -> None:
        """Add the workspace directory to allowedPaths for path-based capabilities.

        The Policy Service returns a static policy bundle where allowedPaths may
        only contain "." (agent process CWD).  The workspace directory is a local
        concern — we enrich the bundle here so that the agent can read/write files
        inside the user's workspace without the backend needing to know local paths.
        """
        from agent_host.policy.path_matcher import resolve_path

        resolved_ws = resolve_path(workspace_dir)
        path_capabilities = {"File.Read", "File.Write", "File.Delete"}

        for cap in policy_bundle.capabilities:
            if cap.name not in path_capabilities:
                continue
            # Build the set of already-resolved allowed paths to avoid duplicates
            existing = {resolve_path(p) for p in cap.allowedPaths} if cap.allowedPaths else set()
            if resolved_ws not in existing:
                if cap.allowedPaths is None:
                    cap.allowedPaths = [resolved_ws]
                else:
                    cap.allowedPaths = [*cap.allowedPaths, resolved_ws]
                logger.debug(
                    "workspace_path_injected",
                    capability=cap.name,
                    workspace_dir=resolved_ws,
                )

    async def start_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """Start a new task (agent work cycle) from a user prompt.

        Spawns the agent loop as an asyncio task and returns immediately.
        """
        if not self._session_context or not self._llm_client:
            raise SessionNotFoundError("No active session")

        task_id = params.get("taskId", "")
        prompt = params.get("prompt", "")

        # Parse taskOptions - extract maxSteps (clamped 1-200)
        task_options = params.get("taskOptions") or {}
        logger.info(
            "start_task_options",
            task_id=task_id,
            plan_only=task_options.get("planOnly"),
            task_options_keys=list(task_options.keys()),
        )
        raw_max_steps = task_options.get("maxSteps", self._config.default_max_steps)
        try:
            max_steps = int(raw_max_steps)
        except (TypeError, ValueError):
            max_steps = self._config.default_max_steps
        max_steps = max(1, min(200, max_steps))
        self._current_max_steps = max_steps
        self._current_step_count = 0

        # Reset cancellation event and sync tracking for new task
        self._cancel_event.clear()
        self._current_task_prompt = prompt
        self._last_workspace_sync_step = 0

        # Check if resuming an interrupted task — skip adding user message
        # since the thread already has it from the checkpoint
        incomplete = self._incomplete_task
        is_resume = incomplete is not None and incomplete.get("taskId") == task_id

        if is_resume and incomplete is not None:
            logger.info(
                "resuming_incomplete_task",
                task_id=task_id,
                last_step=incomplete.get("lastStep", 0),
            )
            self._incomplete_task = None
        else:
            # Prepend workspace context so the LLM always sees it, even after compaction
            llm_prompt = prompt
            if self._workspace_dir:
                llm_prompt = f"[Workspace: {self._workspace_dir}]\n\n{prompt}"

            # In plan-only mode, inject explicit instruction to call CreatePlan
            if task_options.get("planOnly"):
                llm_prompt += (
                    "\n\n[IMPORTANT: You are in plan-only mode. You MUST call the CreatePlan "
                    "tool with a goal and steps before finishing. Use read-only tools to "
                    "explore first, then call CreatePlan. Do NOT skip calling CreatePlan.]"
                )

            # Add user message to thread (with workspace prefix for LLM)
            if self._thread:
                self._thread.add_user_message(llm_prompt)

        # Auto-name session from first user prompt (fire-and-forget)
        if not self._session_messages and prompt and self._session_context:
            auto_name = prompt[:60].strip()
            if len(prompt) > 60:
                auto_name = auto_name.rsplit(" ", 1)[0] + "..."
            asyncio.create_task(self._update_session_name(auto_name))  # noqa: RUF006

        # Report task started to session service (fire-and-forget)
        if self._session_context:
            asyncio.create_task(  # noqa: RUF006
                self._report_task_started(task_id, prompt, max_steps)
            )

        # Emit task_started event
        if self._event_emitter:
            self._event_emitter.emit_task_started(task_id, prompt)

        # Spawn the agent loop as a background asyncio task
        task = asyncio.create_task(self._run_agent(prompt, task_id, max_steps, task_options))
        self._current_task = task
        self._current_task_id = task_id

        return {"taskId": task_id, "status": "running"}

    async def _run_agent(
        self,
        prompt: str,
        task_id: str,
        max_steps: int = 50,
        task_options: dict[str, Any] | None = None,
    ) -> None:
        """Run the agent loop for a single task."""
        if (
            not self._llm_client
            or not self._session_context
            or not self._thread
            or not self._policy_enforcer
            or not self._token_budget
        ):
            return

        task_options = task_options or {}
        plan_only = bool(task_options.get("planOnly", False))
        logger.info(
            "run_task_plan_mode",
            task_id=task_id,
            plan_only=plan_only,
            task_options_keys=list(task_options.keys()),
        )

        assistant_text = ""
        try:
            # Build execution context with workspace working directory
            exec_context: ExecutionContext | None = None
            if self._workspace_dir:
                exec_context = ExecutionContext(working_directory=self._workspace_dir)

            # Build tool executor (with plan mode support)
            tool_executor = ToolExecutor(
                tool_router=self._tool_router,
                policy_enforcer=self._policy_enforcer,
                event_emitter=self._event_emitter,
                approval_gate=self._approval_gate,
                file_change_tracker=self._file_change_tracker,
                workspace_client=self._workspace_client,
                execution_context=exec_context,
                session_id=self._session_context.session_id,
                workspace_id=self._session_context.workspace_id,
                approval_timeout=float(self._config.approval_timeout_seconds),
                plan_mode=plan_only,
                plan_mode_locked=plan_only,
            )

            # Build compactor (hybrid or drop-oldest based on config)
            compactor: ContextCompactor
            if self._config.compaction_strategy == "hybrid":
                compactor = HybridCompactor(
                    recency_window=self._config.recency_window,
                    mask_only=not self._config.compaction_llm_summary,
                )
            else:
                compactor = DropOldestCompactor(
                    recency_window=self._config.recency_window,
                )

            # Working memory (task tracker, plan, notes — injected per-turn)
            if not self._working_memory:
                self._working_memory = WorkingMemory()

            # Plan mode change callback
            def _on_plan_mode_changed(plan_mode: bool, source: str) -> None:
                tool_executor.plan_mode = plan_mode
                if self._event_emitter:
                    self._event_emitter.emit_plan_mode_changed(task_id, plan_mode, source)

            # Emit initial plan mode event for planOnly tasks
            if plan_only and self._event_emitter:
                self._event_emitter.emit_plan_mode_changed(task_id, True, "user")

            # Plan updated callback
            def _on_plan_updated(goal: str, steps: list[dict[str, object]]) -> None:
                if self._event_emitter:
                    self._event_emitter.emit_plan_updated(task_id, goal, steps)

            # Agent-internal tool handler (TaskTracker, CreatePlan, SpawnAgent, memory, skills)
            agent_tool_handler = AgentToolHandler(
                self._working_memory,
                skills=self._skills,
                memory_manager=self._memory_manager,
                on_plan_mode_changed=_on_plan_mode_changed,
                on_plan_updated=_on_plan_updated,
                plan_mode=plan_only,
                plan_mode_locked=plan_only,
            )

            # Build LoopRuntime + ReactLoop
            loop_runtime = LoopRuntime(
                llm_client=self._llm_client,
                tool_executor=tool_executor,
                thread=self._thread,
                compactor=compactor,
                policy_enforcer=self._policy_enforcer,
                token_budget=self._token_budget,
                event_emitter=self._event_emitter,
                cancellation_event=self._cancel_event,
                max_context_tokens=self._max_context_tokens,
                working_memory=self._working_memory,
                memory_manager=self._memory_manager,
                agent_tool_handler=agent_tool_handler,
                on_step_complete=self._on_step_complete,
                skills=self._skills,
            )

            # Build verification config
            skip_verification = bool(task_options.get("skipVerification", False))
            verification: VerificationConfig | None = None
            if self._config.verification_enabled and not skip_verification:
                custom_instructions = task_options.get("verifyInstructions", "")
                # Also check COWORK.md for verification instructions
                if not custom_instructions and self._memory_manager:
                    custom_instructions = self._memory_manager.get_verification_instructions()
                verification = VerificationConfig(
                    enabled=True,
                    max_verify_steps=self._config.verification_max_steps,
                    custom_instructions=custom_instructions,
                )

            strategy = ReactLoop(loop_runtime, max_steps=max_steps, verification=verification)
            result = await strategy.run(task_id)

            # Track step count for get_session_state
            self._current_step_count = result.step_count
            assistant_text = result.text

            # Emit task_completed for successful completion
            if result.reason == "completed" and self._event_emitter:
                self._event_emitter.emit_task_completed(task_id)
                logger.info("task_completed", task_id=task_id)

        except asyncio.CancelledError:
            logger.info("task_cancelled", task_id=task_id)

        except Exception as exc:
            # Classify as session-level or task-level failure
            if isinstance(exc, (PolicyExpiredError, CheckpointError)):
                logger.exception(
                    "session_failed",
                    task_id=task_id,
                    session_id=self._session_context.session_id,
                )
                if self._event_emitter:
                    self._event_emitter.emit_session_failed(str(exc))
            else:
                # Classify LLM errors for structured event payload.
                # Always classify — covers raw transient errors, wrapped
                # LLMGatewayError (retry-exhausted), and unknown errors.
                from agent_host.llm.error_classifier import classify_llm_error

                error_info = classify_llm_error(exc)
                logger.exception(
                    "task_failed",
                    task_id=task_id,
                    session_id=self._session_context.session_id,
                    step_count=self._current_step_count,
                    error_type=error_info.get("error_type", type(exc).__name__),
                )
                if self._event_emitter:
                    self._event_emitter.emit_task_failed(
                        task_id,
                        reason=str(error_info.get("user_message", str(exc))),
                        error_code=str(error_info["error_code"])
                        if error_info.get("error_code")
                        else None,
                        error_type=str(error_info["error_type"])
                        if error_info.get("error_type")
                        else None,
                        is_recoverable=bool(error_info.get("is_recoverable")),
                    )

        finally:
            # Report task completion to session service (best-effort)
            if self._session_context and not isinstance(assistant_text, type(None)):
                completion_status = "completed" if assistant_text else "failed"
                reason = "completed"
                if self._current_step_count >= max_steps:
                    completion_status = "failed"
                    reason = "max_steps_exceeded"
                await self._report_task_completed(
                    task_id, completion_status, self._current_step_count, reason
                )

            # Always upload history and persist checkpoint, even on error/cancel.
            # This ensures the user can see their conversation when they come back.
            await self._upload_history(prompt, assistant_text, task_id)
            self._persist_checkpoint()

    async def _update_session_name(self, name: str) -> None:
        """Update session name on the backend (best-effort, fire-and-forget)."""
        if not self._session_context:
            return
        try:
            await self._session_client.update_session_name(
                self._session_context.session_id, name, auto_named=True
            )
        except Exception:
            logger.warning("auto_name_update_failed", exc_info=True)

    async def _report_task_started(self, task_id: str, prompt: str, max_steps: int) -> None:
        """Report task creation to session service (best-effort)."""
        if not self._session_context:
            return
        try:
            await self._session_client.create_task(
                self._session_context.session_id, task_id, prompt, max_steps
            )
        except Exception:
            logger.warning("task_report_started_failed", task_id=task_id, exc_info=True)

    async def _report_task_completed(
        self,
        task_id: str,
        status: str,
        step_count: int,
        completion_reason: str | None = None,
    ) -> None:
        """Report task completion to session service (best-effort)."""
        if not self._session_context:
            return
        try:
            await self._session_client.complete_task(
                self._session_context.session_id,
                task_id,
                status,
                step_count,
                completion_reason,
            )
        except Exception:
            logger.warning("task_report_completed_failed", task_id=task_id, exc_info=True)

    def _persist_checkpoint(
        self,
        *,
        active_task_id: str | None = None,
        active_task_prompt: str | None = None,
        active_task_step: int = 0,
        active_task_max_steps: int = 0,
        last_workspace_sync_step: int = 0,
    ) -> None:
        """Persist current state into our checkpoint format.

        When called mid-task with active_task_id set, populates in-progress task state.
        When called at task end (no args), clears task state.
        """
        if not self._session_context:
            return

        checkpoint = SessionCheckpoint(
            session_id=self._session_context.session_id,
            workspace_id=self._session_context.workspace_id,
            tenant_id=self._session_context.tenant_id,
            user_id=self._session_context.user_id,
            token_input_used=(self._token_budget.input_tokens_used if self._token_budget else 0),
            token_output_used=(self._token_budget.output_tokens_used if self._token_budget else 0),
            session_messages=[msg.model_dump(mode="json") for msg in self._session_messages],
            thread=self._thread.to_checkpoint() if self._thread else None,
            working_memory=(self._working_memory.to_checkpoint() if self._working_memory else None),
            workspace_dir=self._workspace_dir,
            active_task_id=active_task_id,
            active_task_prompt=active_task_prompt,
            active_task_step=active_task_step,
            active_task_max_steps=active_task_max_steps,
            last_workspace_sync_step=last_workspace_sync_step,
        )
        self._checkpoint_manager.save(checkpoint)

    async def _on_step_complete(self, task_id: str, step: int) -> None:
        """Callback invoked by the loop strategy after each completed step.

        1. Write local checkpoint with active task state (always)
        2. Every N steps, sync history to workspace (best-effort)
        """
        self._current_step_count = step

        # Always write local checkpoint with in-progress task state
        self._persist_checkpoint(
            active_task_id=task_id,
            active_task_prompt=self._current_task_prompt,
            active_task_step=step,
            active_task_max_steps=self._current_max_steps,
            last_workspace_sync_step=self._last_workspace_sync_step,
        )

        if self._event_emitter:
            self._event_emitter.emit_checkpoint_saved(task_id, step)

        # Periodic workspace sync
        sync_interval = self._config.workspace_sync_interval
        if sync_interval > 0 and step - self._last_workspace_sync_step >= sync_interval:
            await self._sync_history_to_workspace(task_id)
            self._last_workspace_sync_step = step
            # Update checkpoint with new sync step
            self._persist_checkpoint(
                active_task_id=task_id,
                active_task_prompt=self._current_task_prompt,
                active_task_step=step,
                active_task_max_steps=self._current_max_steps,
                last_workspace_sync_step=self._last_workspace_sync_step,
            )

    async def _sync_history_to_workspace(self, task_id: str) -> None:
        """Upload cumulative session messages to workspace (best-effort).

        Reuses existing workspace_client.upload_session_history().
        Failures logged, not propagated.
        """
        if not self._session_context:
            return

        try:
            await self._workspace_client.upload_session_history(
                workspace_id=self._session_context.workspace_id,
                session_id=self._session_context.session_id,
                messages=self._session_messages,
                task_id=task_id,
            )
            logger.info("workspace_sync_completed", task_id=task_id)
            if self._event_emitter:
                self._event_emitter.emit_workspace_sync_completed(task_id)
        except Exception:
            logger.warning("workspace_sync_failed", task_id=task_id, exc_info=True)
            if self._event_emitter:
                self._event_emitter.emit_workspace_sync_failed(task_id)

    async def _restore_from_checkpoint(self) -> None:
        """Restore token budget and session messages from checkpoint if available.

        If no local checkpoint exists (corrupt or missing), falls back to
        fetching conversation history from the Workspace Service.
        """
        if not self._session_context:
            return

        checkpoint = self._checkpoint_manager.load(self._session_context.session_id)
        if not checkpoint:
            # Fallback: try workspace service for conversation history
            try:
                messages = await self._workspace_client.get_session_history(
                    workspace_id=self._session_context.workspace_id,
                    session_id=self._session_context.session_id,
                )
                if messages:
                    self._session_messages = messages
                    logger.info(
                        "session_messages_restored_from_workspace",
                        count=len(messages),
                    )
            except Exception:
                logger.warning("workspace_fallback_failed", exc_info=True)
            return

        # Restore workspace_dir (critical for memory + project instructions after crash)
        if checkpoint.workspace_dir and not self._workspace_dir:
            self._workspace_dir = checkpoint.workspace_dir
            logger.info(
                "workspace_dir_restored_from_checkpoint", workspace_dir=checkpoint.workspace_dir
            )

        # Restore token budget
        if self._token_budget and (checkpoint.token_input_used or checkpoint.token_output_used):
            self._token_budget.restore_usage(
                checkpoint.token_input_used, checkpoint.token_output_used
            )
            logger.info(
                "token_budget_restored",
                input_tokens=checkpoint.token_input_used,
                output_tokens=checkpoint.token_output_used,
            )

        # Restore thread (conversation history for the agent loop)
        if checkpoint.thread and self._thread:
            restored_thread = MessageThread.from_checkpoint(checkpoint.thread)
            self._thread = restored_thread
            logger.info(
                "thread_restored_from_checkpoint",
                message_count=restored_thread.message_count,
            )

        # Restore working memory
        if checkpoint.working_memory:
            try:
                self._working_memory = WorkingMemory.from_checkpoint(checkpoint.working_memory)
                logger.info("working_memory_restored_from_checkpoint")
            except Exception:
                logger.warning("working_memory_restore_failed", exc_info=True)

        # Restore session messages
        if checkpoint.session_messages and not self._session_messages:
            try:
                self._session_messages = [
                    ConversationMessage.model_validate(msg) for msg in checkpoint.session_messages
                ]
                logger.info(
                    "session_messages_restored",
                    count=len(self._session_messages),
                )
            except Exception:
                logger.warning("session_messages_restore_failed", exc_info=True)

        # Detect incomplete task from crash
        if checkpoint.active_task_id:
            self._incomplete_task = {
                "taskId": checkpoint.active_task_id,
                "prompt": checkpoint.active_task_prompt or "",
                "lastStep": checkpoint.active_task_step,
                "maxSteps": checkpoint.active_task_max_steps,
            }
            logger.info(
                "incomplete_task_detected",
                task_id=checkpoint.active_task_id,
                last_step=checkpoint.active_task_step,
            )

        if self._event_emitter:
            self._event_emitter.emit_checkpoint_restored(source="local")

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

            # Build tool messages from the thread (current task's messages)
            if self._thread:
                tool_msg_index = 0
                for msg in self._thread.messages:
                    if msg.get("role") == "tool":
                        tool_output = msg.get("content", "")
                        if len(tool_output) > _MAX_TOOL_OUTPUT_LENGTH:
                            tool_output = tool_output[:_MAX_TOOL_OUTPUT_LENGTH] + "... [truncated]"
                        tool_msg_data = {
                            "type": "tool_result",
                            "toolName": msg.get("name", ""),
                            "output": tool_output,
                        }
                        task_messages.append(
                            ConversationMessage(
                                messageId=f"{task_id}-tool-{tool_msg_index}",
                                sessionId=self._session_context.session_id,
                                taskId=task_id,
                                role="tool",
                                content=json.dumps(tool_msg_data, default=str),
                                timestamp=now,
                            )
                        )
                        tool_msg_index += 1
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            fn = tc.get("function", {})
                            tool_call_data = {
                                "type": "tool_call",
                                "toolName": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}"),
                            }
                            task_messages.append(
                                ConversationMessage(
                                    messageId=f"{task_id}-tool-{tool_msg_index}",
                                    sessionId=self._session_context.session_id,
                                    taskId=task_id,
                                    role="tool",
                                    content=json.dumps(tool_call_data, default=str),
                                    timestamp=now,
                                )
                            )
                            tool_msg_index += 1

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
            self._persist_checkpoint()

            logger.info("session_history_uploaded", task_id=task_id)
        except Exception:
            logger.warning("session_history_upload_failed", task_id=task_id, exc_info=True)

    async def cancel_task(self) -> dict[str, Any]:
        """Cancel the currently running task (cooperative).

        Sets the cancellation event instead of task.cancel() for cooperative cancellation.
        """
        if not self._current_task:
            raise NoActiveTaskError("No task is currently running")

        # Signal the agent loop to stop
        self._cancel_event.set()

        # Also cancel the asyncio task as a fallback
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

        if self._incomplete_task:
            state["incompleteTask"] = self._incomplete_task

        return state

    async def shutdown(self) -> dict[str, Any]:
        """Clean session teardown."""
        # Check for active task *before* cancelling it
        had_active_task = bool(self._current_task and not self._current_task.done())

        # Cancel any running task
        if had_active_task and self._current_task:
            self._cancel_event.set()
            self._current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._current_task

        session_id = ""
        if self._session_context:
            session_id = self._session_context.session_id

            # Only cancel session on backend if there was a running task at shutdown
            if had_active_task:
                try:
                    await self._session_client.cancel_session(
                        session_id,
                        SessionCancelRequest(reason="shutdown"),
                    )
                except Exception:
                    logger.warning("shutdown_cancel_failed", exc_info=True)

            # Delete checkpoint (clean exit)
            self._checkpoint_manager.delete(session_id)

        # Emit session_completed before cleanup
        if self._event_emitter:
            self._event_emitter.emit_session_completed()

        # Close LLM client
        if self._llm_client:
            await self._llm_client.close()

        # Close HTTP clients
        await self._session_client.close()
        await self._workspace_client.close()

        self._session_context = None
        self._llm_client = None
        self._current_task = None
        self._approval_gate = None
        self._file_change_tracker = None
        self._thread = None
        self._working_memory = None
        self._memory_manager = None

        logger.info("session_shutdown", session_id=session_id)
        return {"status": "shutdown", "sessionId": session_id}
