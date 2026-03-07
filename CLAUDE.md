# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

`cowork-agent-runtime` contains the Local Agent Host and Local Tool Runtime — the Python process that runs the agent loop on the user's desktop. It is spawned by the Desktop App as a child process and communicates via JSON-RPC 2.0 over stdio.

## Architecture

Two top-level packages with a **strict boundary** — no cross-imports allowed:

```
agent_host/     ← Local Agent Host (custom agent loop)
  server/       — JSON-RPC 2.0 server over stdio (parse, serialize, dispatch, handlers)
  session/      — Session/Workspace HTTP clients (tenacity retry), checkpoint manager, SessionManager
  loop/         — LoopRuntime (infrastructure), LoopStrategy protocol, ReactLoop (default strategy), tool executor, agent-internal tools, error recovery
  llm/          — LLM Gateway streaming client (openai SDK), response models, error classifier
  thread/       — Message thread management, context compaction, token counting
  memory/       — Working memory: task tracker, plan, notes (injected per-turn)
  skills/       — Skill definitions, loader (built-in/markdown/policy); execution via LoopRuntime
  policy/       — Policy Enforcer: capability validation, path/command/domain matchers, risk assessor
  budget/       — Token budget tracking (pre-check + record_usage)
  approval/     — Approval gate (asyncio Futures for user approval flow)
  events/       — Event emitter: SessionEvent notifications + structured logging

tool_runtime/   ← Local Tool Runtime (tool execution)
  router/       — ToolRouter implementation, tool registry, dispatch
  tools/
    file/       — ReadFile, WriteFile, DeleteFile, EditFile, MultiEdit, CreateDirectory, MoveFile, ListDirectory, FindFiles, GrepFiles, ViewImage
    shell/      — RunCommand
    network/    — HttpRequest, FetchUrl, WebSearch
    code/       — ExecuteCode (Python script execution)
  code/         — Code execution engine (PythonExecutor, preamble, CodeExecutionResult)
  platform/     — OS abstraction (path handling, shell resolution, encoding) for macOS/Windows
  mcp/          — MCP client: discovery, connection, manifest translation (Phase 2+)
  output/       — Output formatting, truncation, artifact extraction
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
  - `LoopStrategy` protocol (`loop/strategy.py`) — single method `async def run(task_id) -> LoopResult`. Strategies compose LoopRuntime primitives.
  - `ReactLoop` (`loop/react_loop.py`) — default strategy (linear ReAct). Owns context assembly (memory injection, working memory, compaction, error recovery) and tool routing (agent-internal vs external).
  - `AgentLoop` (`loop/agent_loop.py`) — thin alias for `ReactLoop` (backward compat).
- **OpenAI SDK** (`openai.AsyncOpenAI`) for streaming to LLM Gateway's OpenAI-compatible endpoint.
- **Infrastructure layers inside LoopRuntime:**
  - `ToolExecutor` — policy check → approval gate → file change tracking → ToolRouter dispatch → artifact upload
  - `AgentToolHandler` — routes agent-internal tools (TaskTracker, CreatePlan, SpawnAgent, memory, skills) without going through PolicyEnforcer. Uses callbacks to LoopRuntime for sub-agent/skill execution.
  - `ErrorRecovery` — consecutive failure tracking, loop detection (same tool+args 3+ times), reflection/loop-break prompt injection
  - `WorkingMemory` — task tracker + plan + notes, injected as system message every turn (by ReactLoop)
  - Sub-agent spawning — `LoopRuntime.spawn_sub_agent()` creates child LoopRuntime + ReactLoop with isolated MessageThread, shared TokenBudget, Semaphore(5) concurrency
  - Skill execution — `LoopRuntime.execute_skill()` runs skills as focused sub-conversations with child LoopRuntime + ReactLoop
- **Context compaction** (`thread/compactor.py`) — drop-oldest with recency window, triggered at 90% of max_context_tokens.
- **Custom JSON-RPC 2.0 server** (~200 lines). Newline-delimited JSON with write lock.
- **CheckpointManager** — atomic JSON file writes (tempfile + os.replace) for crash recovery. Persists thread, token budget, working memory.
- **Policy Enforcer** is pure — no I/O, no async. Receives `PolicyBundle` at init, indexes capabilities by name.
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
      __init__.py
      exceptions.py           # AgentHostError hierarchy → JSON-RPC error codes
      models.py               # SessionContext, PolicyCheckResult
      config.py               # AgentHostConfig from env vars
      main.py                 # Process entry point
      server/                 # JSON-RPC 2.0 server (parse, transport, dispatch, handlers)
      session/                # Session/Workspace clients, checkpoint manager, SessionManager
      loop/                   # Agent loop, tool executor, agent tools, error recovery, sub-agents
      llm/                    # LLM Gateway streaming client, response models, error classifier
      thread/                 # Message thread, context compaction, token counting
      memory/                 # Working memory: task tracker, plan, notes
      skills/                 # Skill definitions, loader, executor
      policy/                 # Policy enforcer, path/command/domain matchers, risk assessor
      budget/                 # Token budget tracking
      approval/               # Approval gate (asyncio Futures)
      events/                 # Event emitter
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
  └── NoActiveTaskError (-32041)
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
