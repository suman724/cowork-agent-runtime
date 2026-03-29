# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

`cowork-agent-runtime` contains the Local Agent Host and Local Tool Runtime — the Python process that runs the agent loop. It supports two transport modes:
- **stdio** (default): Spawned by the Desktop App as a child process, communicates via JSON-RPC 2.0 over stdin/stdout.
- **http**: Runs as an HTTP/SSE server for web/sandbox mode (`--transport http`). Exposes JSON-RPC via `POST /rpc`, events via `GET /events` (SSE), and file operations via `/upload` and `/files`.

## Architecture

Three packages: `agent_host/` and `tool_runtime/` (strict boundary, no cross-imports), plus `cowork-agent-sdk` (external dependency providing reusable agent primitives).

```
agent_host/     ← Local Agent Host (cowork application layer)
  transport/    — Transport protocol, StdioTransport, HttpTransport, JSON-RPC 2.0, MethodDispatcher
  server/       — JSON-RPC method handlers (thin delegation to SessionManager)
  session/      — Session/Workspace HTTP clients (tenacity retry), SessionManager
  loop/         — LoopRuntime (infrastructure facade), tool executor, agent-internal tools, sub-agents
  approval/     — ApprovalClient (HTTP client to Approval Service)
  events/       — EventEmitter, EventBuffer (SSE replay ring buffer)
  sandbox/      — Sandbox mode: self-registration (startup.py), workspace file sync (workspace_sync.py)

tool_runtime/   ← Local Tool Runtime (tool execution, unchanged)
  router/       — ToolRouter implementation, tool registry, dispatch
  tools/
    file/       — ReadFile, WriteFile, DeleteFile, EditFile, MultiEdit, CreateDirectory, MoveFile, ListDirectory, FindFiles, GrepFiles, ViewImage
    shell/      — RunCommand
    network/    — HttpRequest, FetchUrl, WebSearch
    code/       — ExecuteCode (Python script execution)
    browser/    — BrowserNavigate, BrowserClick, BrowserType, BrowserSelect, BrowserScroll, BrowserBack, BrowserExtract, BrowserScreenshot, BrowserSubmit, BrowserDownload, BrowserWait (Playwright, desktop-only, opt-in)
  code/         — Code execution engine (PythonExecutor, preamble, CodeExecutionResult)
  platform/     — OS abstraction (path handling, shell resolution, encoding) for macOS/Windows
  mcp/          — MCP client: discovery, connection, manifest translation (Phase 2+)
  output/       — Output formatting, truncation, artifact extraction

# External dependency (pip-installed from cowork-agent-sdk repo):
agent_sdk/      ← Reusable agent building blocks
  loop/         — LoopContext protocol, LoopStrategy protocol, ReactLoop, error recovery, verification
  thread/       — MessageThread, context compaction (DropOldest, Hybrid), token counting
  memory/       — WorkingMemory, MemoryManager, persistent memory, plan, task tracker
  policy/       — PolicyEnforcer (pure, no I/O): capability validation, path/command/domain matchers
  llm/          — LLM Gateway streaming client (openai SDK), response models, error classifier
  budget/       — TokenBudget tracking (pre-check + record_usage)
  approval/     — ApprovalGate mechanism (asyncio Futures)
  skills/       — SkillLoader, SkillDefinition (discovery & loading)
  checkpoint/   — CheckpointManager (crash recovery persistence)
  tracking/     — FileChangeTracker (file mutation tracking for patch preview)
```

**Communication boundary:** `agent_host/` calls `tool_runtime/` only through `ToolRouter` and `ExecutionContext`:
```python
from tool_runtime import ToolRouter, ExecutionContext, ToolExecutionResult
# ToolRouter.execute(request: ToolRequest, context: ExecutionContext | None) -> ToolExecutionResult
# ToolRouter.get_available_tools() -> list[ToolDefinition]
```

## Key Patterns

- **Three-layer agent loop architecture** — `SessionManager` (session lifecycle) → `LoopRuntime` (per-task infrastructure) → `LoopStrategy` (orchestration + context assembly). See `cowork-infra/docs/components/loop-strategy.md`.
  - `LoopRuntime` (`loop/loop_runtime.py`) — infrastructure primitives facade. Provides `call_llm()`, `execute_external_tools()`, `execute_agent_tool()`, `spawn_sub_agent()`, `execute_skill()`, event emission, checkpoint callbacks, token budget. Owns all backend service coupling.
  - `LoopContext` protocol (`agent_sdk/loop/context.py`) — defines the interface between loop strategies and `LoopRuntime`. `LoopRuntime` implements `LoopContext`.
  - `LoopStrategy` protocol (`agent_sdk/loop/strategy.py`) — single method `async def run(task_id) -> LoopResult`. Strategies compose LoopContext primitives.
  - `ReactLoop` (`agent_sdk/loop/react_loop.py`) — default strategy (linear ReAct). Owns context assembly (memory injection, working memory, compaction, error recovery) and tool routing (agent-internal vs external). Depends on `LoopContext`, not concrete `LoopRuntime`.
  - `AgentLoop` (`loop/agent_loop.py`) — thin alias for `ReactLoop` (backward compat).
- **OpenAI SDK** (`openai.AsyncOpenAI`) for streaming to LLM Gateway's OpenAI-compatible endpoint.
- **Infrastructure layers inside LoopRuntime:**
  - `ToolExecutor` — policy check → approval gate → file change tracking → ToolRouter dispatch → artifact upload. Supports **parallel tool execution** via `asyncio.gather()` with intelligent grouping (read-only tools batched, writes serialized per path, shell commands always serial). Also enforces **plan mode** restrictions (filters tool definitions, denies blocked tools with `PLAN_MODE_RESTRICTED`).
  - `AgentToolHandler` — routes agent-internal tools (TaskTracker, CreatePlan, **UpdatePlanStep**, SpawnAgent, memory, skills, **EnterPlanMode**, **ExitPlanMode**) without going through PolicyEnforcer. Uses callbacks to LoopRuntime for sub-agent/skill execution. `UpdatePlanStep` marks plan steps as `in_progress`/`completed`/`skipped` and triggers the `plan_updated` event via `_notify_plan_updated()` → `SessionManager._on_plan_updated()` → `EventEmitter.emit_plan_updated()`.
  - `ErrorRecovery` (`agent_sdk/loop/error_recovery.py`) — consecutive failure tracking, loop detection (same tool+args 3+ times), reflection/loop-break prompt injection
  - `WorkingMemory` (`agent_sdk/memory/working_memory.py`) — task tracker + plan + notes, injected as system message every turn (by ReactLoop)
  - `VerificationConfig` (`agent_sdk/loop/verification.py`) — post-completion self-verification. Injects verification prompt when agent first signals done, extends step budget by `max_verify_steps`, emits `verification_started`/`verification_completed` events.
  - Sub-agent spawning — `LoopRuntime.spawn_sub_agent()` creates child LoopRuntime + ReactLoop with isolated MessageThread, shared TokenBudget, Semaphore(5) concurrency
  - Skill execution — `LoopRuntime.execute_skill()` runs skills as focused sub-conversations with child LoopRuntime + ReactLoop
- **Context compaction** (in `agent_sdk.thread.compactor`) — two strategies: `DropOldestCompactor` (simple drop with recency window) and `HybridCompactor` (observation masking + optional LLM summarization). Default: `hybrid`. Triggered at 90% of max_context_tokens.
- **Prompt caching optimization** — `ReactLoop._build_messages()` orders context for LLM provider cache efficiency: stable prefix (system prompt → persistent memory → conversation history) then volatile suffix (working memory → error recovery).
- **Transport protocol** (`transport/transport.py`) — `Transport` protocol with `start()`, `send_event()`, `shutdown()`. Two implementations:
  - `StdioTransport` (`transport/stdio_transport.py`) — JSON-RPC over stdin/stdout with write lock (desktop mode)
  - `HttpTransport` (`transport/http_transport.py`) — Starlette/uvicorn ASGI server (sandbox/web mode): `POST /rpc`, `GET /events` (SSE with replay), `GET /health`, `GET /ready`, `POST /upload`, `GET /files/{path}`, `GET /files`
- **Shared EventBuffer** (`events/event_buffer.py`) — bounded ring buffer (default 10K events) with monotonic IDs. Owned by `EventEmitter`, shared with both transports. Enables:
  - SSE replay via `?since={id}` for HttpTransport
  - `GetEvents` JSON-RPC method for Desktop App event replay after view navigation
  - All `SessionEvent` notifications include `eventId` for client-side tracking
- **Custom JSON-RPC 2.0 protocol** (`transport/json_rpc.py`, ~200 lines). Shared by both transports via `MethodDispatcher` (`transport/method_dispatcher.py`).
- **CheckpointManager** (`agent_sdk/checkpoint/`) — atomic JSON file writes (tempfile + os.replace) for crash recovery. Persists thread, token budget, working memory.
- **PolicyEnforcer** (`agent_sdk/policy/`) is pure — no I/O, no async. Receives `PolicyBundle` at init, indexes capabilities by name.
- **Pydantic models** from `cowork-platform` for all data contracts.
- **httpx** with `tenacity` retry for async HTTP to backend services.
- **structlog** to stderr for structured logging; stdout reserved for JSON-RPC.

## Tool-to-Capability Mapping

| Tool | Capability | Description |
|------|-----------|-------------|
| `ReadFile` | `File.Read` | Read file contents with encoding detection |
| `WriteFile` | `File.Write` | Atomic file write with diff generation |
| `DeleteFile` | `File.Delete` | Delete a file |
| `EditFile` | `File.Write` | Exact-match find-and-replace editing |
| `MultiEdit` | `File.Write` | Batch multiple find-and-replace edits atomically |
| `CreateDirectory` | `File.Write` | Create directories without shell commands |
| `MoveFile` | `File.Write` | Move or rename files and directories |
| `ListDirectory` | `File.Read` | List files and directories at a path |
| `FindFiles` | `File.Read` | Glob-pattern file discovery across a directory tree |
| `GrepFiles` | `File.Read` | Regex search across files |
| `ViewImage` | `File.Read` | Read image file, return base64 for multimodal LLM |
| `RunCommand` | `Shell.Exec` | Execute shell commands (requires description) |
| `HttpRequest` | `Network.Http` | General HTTP requests |
| `FetchUrl` | `Network.Http` | Fetch URL, convert HTML→markdown |
| `WebSearch` | `Search.Web` | Web search via Tavily API |
| `ExecuteCode` | `Code.Execute` | Execute Python scripts with output capture and matplotlib support |
| `BrowserNavigate` | `Browser.Navigate` | Navigate headed browser to URL with SSRF prevention |
| `BrowserClick` | `Browser.Interact` | Click interactive element by index with sensitive detection |
| `BrowserType` | `Browser.Interact` | Type into input field by index with sensitive field detection |
| `BrowserSelect` | `Browser.Interact` | Select dropdown option, checkbox, or radio button |
| `BrowserScroll` | `Browser.Navigate` | Scroll page with lazy-load wait |
| `BrowserBack` | `Browser.Navigate` | Navigate browser history back |
| `BrowserExtract` | `Browser.Extract` | Read page content as markdown/text/HTML |
| `BrowserScreenshot` | `Browser.Extract` | Capture viewport/full-page/element screenshot |
| `BrowserSubmit` | `Browser.Submit` | Submit form with mandatory approval checkpoint |
| `BrowserDownload` | `Browser.Download` | Download file to workspace with approval |
| `BrowserWait` | `Browser.Navigate` | Wait for element, navigation, or network idle |

## Environment Variables

- `LLM_GATEWAY_ENDPOINT` — LLM Gateway URL (required)
- `LLM_GATEWAY_AUTH_TOKEN` — LLM Gateway auth token (required)
- `SESSION_SERVICE_URL` — Session Service URL (required)
- `WORKSPACE_SERVICE_URL` — Workspace Service URL (required)
- `CHECKPOINT_DIR` — Checkpoint directory (default: platform app data)
- `APPROVAL_TIMEOUT_SECONDS` — Approval timeout (default: 300)
- `LOG_LEVEL` — Logging level (default: info)
- `LLM_MODEL` — LLM model identifier (default: openai/gpt-4o)
- `TAVILY_API_KEY` — Tavily API key (optional, required for WebSearch tool)
- `WORKSPACE_SYNC_INTERVAL` — Sync checkpoint to workspace every N steps (default: 5, 0 = disabled)
- `SESSION_ID` — Pre-assigned session ID (legacy sandbox mode, triggers self-registration)
- `REGISTRATION_TOKEN` — Token for sandbox self-registration (legacy sandbox mode)
- `SANDBOX_LOCAL_MODE` — Skip ECS metadata, use localhost endpoint (sandbox local dev)
- `SKILLS_DIR` — Override user skills directory (default: `~/.cowork/skills/`)
- `SQS_QUEUE_URL` — SQS queue for session dispatch (SQS sandbox mode, overrides SESSION_ID)
- `AWS_ENDPOINT_URL` — AWS endpoint override for LocalStack (e.g., `http://localhost:4566`)
- `SANDBOX_SERVICE_NAME` — CloudWatch metric dimension (default: `sandbox-workers`)
- `ENVIRONMENT` — Environment name for CloudWatch metrics (default: `dev`)

## Sandbox Mode

The agent runtime supports two sandbox config sources with `--transport http`:

1. **SQS mode** (`SQS_QUEUE_URL` set): Polls SQS for session config, picks up a session, serves it, then exits. Used in production ECS Service worker pool. See `cowork-infra/docs/design/sqs-sandbox-dispatch.md`.
2. **Legacy env var mode** (`SESSION_ID` set, no `SQS_QUEUE_URL`): Reads session config from environment variables. Useful for debugging and manual sandbox start.

In either mode, the sandbox startup flow is:

1. **Self-registration**: Reads container IP from ECS metadata (or localhost in `SANDBOX_LOCAL_MODE`), calls `POST /sessions/{sessionId}/register` on Session Service
2. **Workspace sync**: Downloads workspace files from Workspace Service to `--workspace-dir` before serving HTTP. Sets the startup sync gate (`asyncio.Event`) after completion.
3. **Session initialization**: Initializes from registration response (policy bundle, workspace ID) — skips `CreateSession` RPC. Loads prior session history from Workspace Service (no-op for new sessions, restores full conversation thread for resumed sessions).
4. **Skills**: Loads from `{workspace}/.cowork/skills/` (project-level) in addition to built-in skills. No home directory needed. `SKILLS_DIR` env var overrides the user skills path.
5. **Graceful shutdown**: On SIGTERM, uploads workspace files back to Workspace Service before exiting
6. **`workspace.sync` RPC** (HTTP transport only): Session Service can trigger targeted file sync via `POST /rpc` with method `workspace.sync`. Supports `direction` (`pull`/`push`) and optional `paths` list. Serialized via `asyncio.Lock`, gated behind startup sync completion (30s timeout). See `workspace-file-sync.md` design doc.
7. **CloudWatch metrics** (SQS mode only): Publishes `TaskUtilization` metric (1.0 = busy, 0.0 = idle) to `Cowork/Sandbox` namespace. Used for ECS auto-scaling. Best-effort — no-op if CloudWatch unavailable.
8. **Process exit** (SQS mode only): After session ends and workspace sync completes, the process exits. ECS replaces it with a fresh container.

Stdio mode is completely unaffected by sandbox-related code.

## CLI Arguments

- `--transport {stdio,http}` — Transport mode (default: `stdio`)
- `--host HOST` — HTTP server bind address (default: `0.0.0.0`, only with `--transport http`)
- `--port PORT` — HTTP server port (default: `8080`, only with `--transport http`)
- `--workspace-dir DIR` — Workspace directory for file upload/download (only with `--transport http`)

## Platform Adapters

`tool_runtime/platform/` abstracts macOS vs Windows differences:
- Path separators, case sensitivity, max length, symlink resolution
- Shell resolution (`/bin/zsh` vs `cmd.exe`), process tree kill signals
- Encoding fallback chain: utf-8 → OS default → latin-1

## Output Truncation

When tool output exceeds `maxOutputBytes`: keep first 80% (head) + last 20% (tail) with a marker between. Outputs >10KB become artifacts uploaded to Workspace Service; the LLM sees the truncated version.

## Design Doc

Full specification: `cowork-infra/docs/components/local-agent-host.md` and `cowork-infra/docs/components/local-tool-runtime.md`

---

## Engineering Standards

### Project Structure

```
cowork-agent-runtime/
  CLAUDE.md
  README.md
  Makefile
  pyproject.toml
  .python-version             # 3.12
  .env.example
  src/
    agent_host/
      __init__.py             # Re-exports AgentHostError, SessionContext, PolicyCheckResult from agent_sdk
      config.py               # AgentHostConfig from env vars
      main.py                 # Process entry point
      transport/              # Transport protocol, StdioTransport, HttpTransport, JSON-RPC, MethodDispatcher
      server/                 # JSON-RPC method handlers (thin delegation to SessionManager)
      session/                # Session/Workspace HTTP clients, SessionManager
      loop/                   # LoopRuntime (implements LoopContext), tool executor, agent tools, sub-agents
      approval/               # ApprovalClient (HTTP to Approval Service)
      events/                 # EventEmitter, EventBuffer (SSE replay)
      sandbox/                # Sandbox startup (self-registration), workspace file sync
    tool_runtime/
      __init__.py
      router/                 # ToolRouter implementation
      tools/
        file/                 # ReadFile, WriteFile, DeleteFile, EditFile, MultiEdit, CreateDirectory, MoveFile, ListDirectory, FindFiles, GrepFiles, ViewImage
        shell/                # RunCommand
        network/              # HttpRequest, FetchUrl, WebSearch
        code/                 # ExecuteCode
      code/                   # PythonExecutor, preamble, CodeExecutionResult
      platform/               # OS abstraction (macOS/Windows)
      mcp/                    # MCP client (Phase 2+)
      output/                 # Formatting, truncation, artifact extraction
  tests/
    unit/
      agent_host/             # Mirrors src/agent_host/ structure
      tool_runtime/           # Mirrors src/tool_runtime/ structure
    integration/              # End-to-end agent loop tests
    fixtures/                 # Shared test data (policy bundles, mock LLM, tool requests)
    conftest.py
  build/                      # Platform-specific packaging (Phase 4)
```

### Python Tooling

- **Python**: 3.12+
- **Linting/formatting**: `ruff`
  - Enable rule sets: `E`, `F`, `W`, `I`, `N`, `UP`, `S`, `B`, `A`, `C4`, `SIM`, `TCH`, `ARG`, `PTH`, `RUF`
  - Line length: 100
  - `S` (bandit) rules are critical here — this code executes shell commands and file operations
- **Type checking**: `mypy --strict`
- **Testing**: `pytest` with `pytest-asyncio`
- **Coverage**: 90% combined for agent_host/ + tool_runtime/

### Dependencies

| Library | Purpose |
|---------|---------|
| `openai>=1.60,<2.0` | LLM Gateway streaming client (AsyncOpenAI, OpenAI-compatible endpoint) |
| `tenacity>=9.0,<10.0` | Retry with exponential backoff for HTTP clients |
| `httpx>=0.27,<1.0` | Async HTTP for backend service calls |
| `pydantic>=2.0,<3.0` | Data validation (from cowork-platform contracts) |
| `structlog>=24.0,<26.0` | Structured logging to stderr |
| `markdownify>=0.14,<1.0` | HTML to markdown conversion (FetchUrl tool) |
| `starlette>=0.41,<1.0` | ASGI framework for HttpTransport (web/sandbox mode) |
| `uvicorn>=0.32,<1.0` | ASGI server for HttpTransport |
| `python-multipart>=0.0.18,<1.0` | Multipart form parsing for file upload |

### Package Boundary Enforcement

**`agent_host/` and `tool_runtime/` must NEVER cross-import.** The only interface between them is `ToolRouter` and `ExecutionContext`. Enforce with:
- Separate `__init__.py` exports — `tool_runtime` exports only `ToolRouter`, `ExecutionContext`, `ToolExecutionResult`
- Test structure mirrors the package boundary — separate test directories

### Makefile Targets

```
make help                # Show all targets
make install             # Install dependencies (pip install -e ".[dev]")
make lint                # Run ruff check
make format              # Run ruff format
make format-check        # Check formatting
make typecheck           # Run mypy --strict
make test                # Run all unit tests
make test-integration    # Integration tests (pytest -m integration)
make coverage            # Run tests with coverage report
make check               # CI gate: lint + format-check + typecheck + test
make clean               # Remove build artifacts and caches
```

### Error Handling

Custom exception hierarchy mapping to JSON-RPC error codes:
```
AgentHostError (base, json_rpc_code=-32000)
  ├── SessionNotFoundError (-32001)
  ├── SessionExpiredError (-32002)
  ├── PolicyExpiredError (-32003)
  ├── LLMGatewayError (-32010)
  ├── LLMBudgetExceededError (-32011)
  ├── LLMGuardrailBlockedError (-32012)
  ├── CapabilityDeniedError (-32020)
  ├── ApprovalRequiredError (-32021, carries approval_rule_id + risk_level)
  ├── ApprovalTimeoutError (-32022)
  ├── CheckpointError (-32030)
  ├── TaskCancelledError (-32040)
  ├── NoActiveTaskError (-32041)
  ├── SandboxStartupError (-32060)
  └── WorkspaceSyncError (-32061)
```

All exceptions carry structured context for logging. The `MethodDispatcher` catches `AgentHostError` and maps `json_rpc_code` to JSON-RPC error responses; unexpected exceptions become `-32603 Internal error`.

### Async Patterns

- **All I/O is async**: `async def` for every function that does I/O (file, network, subprocess).
- **httpx.AsyncClient** with connection pooling for all outbound HTTP. Create one client per session, close on shutdown.
- **LLM streaming**: `LLMClient.stream_chat()` uses `openai.AsyncOpenAI` with text chunk callbacks. Retry with backoff via `error_classifier.py`.
- **Never use `time.sleep()`** — use `asyncio.sleep()`.
- **Background tasks**: `asyncio.create_task()` for agent loop execution from `start_task()`.

### Subprocess Management (RunCommand)

- Use `asyncio.create_subprocess_exec` (not `subprocess.Popen`).
- Capture stdout and stderr via separate `StreamReader` instances.
- Timeout: `asyncio.wait_for` with configurable timeout (default 300s).
- Process tree kill on timeout: send SIGTERM to process group, wait 5s, then SIGKILL.
- Platform-specific kill: `os.killpg` on macOS, `taskkill /T /F` on Windows — handled by `platform/` module.

### Checkpoint / Recovery

- JSON file per session in user's app data directory (atomic write via tempfile + os.replace).
- `CheckpointManager` saves `SessionCheckpoint` with thread state, token budget, working memory.
- Write checkpoint after each step completion.
- On clean session end (Shutdown): delete checkpoint file.
- On crash recovery: load checkpoint, restore thread + token budget + working memory.

### Testing

- **Pytest markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.platform`
- **Agent loop tests**: `MockLLMClient` + mock ToolExecutor → test loop termination, tool dispatch, compaction, cancellation, max_steps
- **Tool tests**: Real filesystem operations in temp directories, real subprocess execution with simple commands
- **Policy enforcer tests**: Various policy bundles × tool requests → verify allow/deny/approval-required
- **Session manager tests**: Mock HTTP clients + mock LLM → test lifecycle
- **Platform tests**: `@pytest.mark.skipif(sys.platform != 'darwin')` for macOS-specific, similar for Windows
- **Fixtures**: Pre-built policy bundles in `tests/fixtures/policy_bundles.py`
- **Sandbox E2E test**: `make test-sandbox` runs the full web sandbox lifecycle test from `cowork-session-service/scripts/test-web-sandbox.py`. Requires LocalStack, backend services, and agent-runtime in HTTP mode (`make run-sandbox`).
