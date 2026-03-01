# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

`cowork-agent-runtime` contains the Local Agent Host and Local Tool Runtime — the Python process that runs the agent loop on the user's desktop. It is spawned by the Desktop App as a child process and communicates via JSON-RPC 2.0 over stdio.

## Architecture

Two top-level packages with a **strict boundary** — no cross-imports allowed:

```
agent_host/     ← Local Agent Host (agent loop orchestration)
  server/       — JSON-RPC 2.0 server (stdio Phase 1, local socket Phase 2+)
  session/      — Session client (calls Session Service over HTTPS)
  loop/         — Core agent loop: plan → LLM call → tool exec → checkpoint → repeat
  routing/      — Tool routing: capability check → approval gate → dispatch to ToolRouter
  policy/       — Local Policy Enforcer: capability validation, path/command restrictions
  llm/          — LLM Gateway client (HTTP streaming via httpx)
  state/        — Local State Store: SQLite checkpoint per step for crash recovery
  events/       — Event emitter: SessionEvent notifications to Desktop App + audit/telemetry

tool_runtime/   ← Local Tool Runtime (tool execution)
  router/       — ToolRouter implementation, tool registry, dispatch
  tools/
    file/       — ReadFile, WriteFile, DeleteFile
    shell/      — RunCommand
    network/    — HttpRequest
  platform/     — OS abstraction (path handling, shell resolution, encoding) for macOS/Windows
  mcp/          — MCP client: discovery, connection, manifest translation (Phase 2+)
  output/       — Output formatting, truncation, artifact extraction
```

**Communication boundary:** `agent_host/` calls `tool_runtime/` only through the `ToolRouter` interface:
```python
interface ToolRouter:
    execute(request: ToolRequest) -> ToolResult
    getAvailableTools() -> list[ToolDefinition]
```

## Key Patterns

- **Python asyncio** throughout — the agent loop is I/O-bound (LLM streaming, file ops, shell commands).
- **Pydantic models** from `cowork-platform` for all data contracts (ToolRequest, ToolResult, PolicyBundle, etc.).
- **httpx** for async HTTP (LLM Gateway streaming, backend service calls).
- Agent loop per step: LLM call → parse tool calls → for each tool call: policy check → approval if needed → execute → collect result → checkpoint → next step.
- **Local Policy Enforcer** validates every tool call against the policy bundle before execution (capabilities, allowedPaths, allowedCommands, allowedDomains).
- **Local State Store** writes a SQLite checkpoint after each step. Cleared on clean session end. Used only for crash recovery.
- **Message thread** is held in memory for the session lifetime. Uploaded to Workspace Service as `session_history` artifact after each task.

## Tool-to-Capability Mapping

| Tool | Capability |
|------|-----------|
| `ReadFile` | `File.Read` |
| `WriteFile` | `File.Write` |
| `DeleteFile` | `File.Delete` |
| `RunCommand` | `Shell.Exec` |
| `HttpRequest` | `Network.Http` |

## Platform Adapters

`tool_runtime/platform/` abstracts macOS vs Windows differences:
- Path separators, case sensitivity, max length, symlink resolution
- Shell resolution (`/bin/zsh` vs `cmd.exe`), process tree kill signals
- Encoding fallback chain: utf-8 → OS default → latin-1

## Output Truncation

When tool output exceeds `maxOutputBytes`: keep first 80% (head) + last 20% (tail) with a marker between. Outputs >10KB become artifacts uploaded to Workspace Service; the LLM sees the truncated version.

## Environment Variables

- `LLM_GATEWAY_ENDPOINT` — LLM Gateway URL
- `LLM_GATEWAY_AUTH_TOKEN` — LLM Gateway auth token
- These are NOT received from the Session Service — read from local env only.

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
      server/                 # JSON-RPC 2.0 server
      session/                # Session Service client
      loop/                   # Core agent loop
      routing/                # Tool routing + capability check + approval gate
      policy/                 # Local Policy Enforcer
      llm/                    # LLM Gateway streaming client
      state/                  # Local State Store (SQLite)
      events/                 # Event emitter
    tool_runtime/
      __init__.py
      router/                 # ToolRouter implementation
      tools/
        file/                 # ReadFile, WriteFile, DeleteFile
        shell/                # RunCommand
        network/              # HttpRequest
      platform/               # OS abstraction (macOS/Windows)
      mcp/                    # MCP client (Phase 2+)
      output/                 # Formatting, truncation, artifact extraction
  tests/
    unit/
      agent_host/             # Mirrors src/agent_host/ structure
      tool_runtime/           # Mirrors src/tool_runtime/ structure
    integration/              # End-to-end agent loop tests
    fixtures/                 # Shared test data (policy bundles, tool requests)
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
- **Coverage**: 90% for agent_host/, 90% for tool_runtime/

### Package Boundary Enforcement

**`agent_host/` and `tool_runtime/` must NEVER cross-import.** The only interface between them is `ToolRouter` (Protocol class). Enforce with:
- Separate `__init__.py` exports — `tool_runtime` exports only `ToolRouter` implementation and `ToolDefinition`
- CI check: `ruff` rule or custom lint to detect cross-package imports
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
make test-unit           # Unit tests only (pytest -m unit)
make test-integration    # Integration tests (pytest -m integration)
make test-platform       # Platform-specific tests (macOS or Windows)
make coverage            # Run tests with coverage report
make check               # CI gate: lint + format-check + typecheck + test
make clean               # Remove build artifacts and caches
```

### Error Handling

Custom exception hierarchy:
```
AgentRuntimeError (base)
  ├── SessionError
  │     ├── SessionNotFoundError
  │     ├── SessionExpiredError
  │     └── PolicyExpiredError
  ├── ToolExecutionError
  │     ├── ToolNotFoundError
  │     ├── ToolTimeoutError
  │     └── ToolPermissionError
  ├── LLMError
  │     ├── LLMGatewayError
  │     ├── LLMBudgetExceededError
  │     └── LLMGuardrailBlockedError
  └── PolicyViolationError
        ├── CapabilityDeniedError
        └── ApprovalRequiredError
```

All exceptions carry structured context (session_id, tool_name, etc.) for logging. Map to JSON-RPC error codes when returning to the Desktop App.

### Async Patterns

- **All I/O is async**: `async def` for every function that does I/O (file, network, subprocess).
- **httpx.AsyncClient** with connection pooling for all outbound HTTP (Session Service, Workspace Service, LLM Gateway). Create one client per session, close on session end.
- **LLM streaming**: `httpx.AsyncClient.stream()` with SSE parsing. Yield chunks to the event emitter for real-time UI updates.
- **Never use `time.sleep()`** — use `asyncio.sleep()`.
- **Structured concurrency**: Use `asyncio.TaskGroup` for parallel tool executions within a step (when multiple tool calls are independent).

### Subprocess Management (RunCommand)

- Use `asyncio.create_subprocess_exec` (not `subprocess.Popen`).
- Capture stdout and stderr via separate `StreamReader` instances.
- Timeout: `asyncio.wait_for` with configurable timeout (default 300s).
- Process tree kill on timeout: send SIGTERM to process group, wait 5s, then SIGKILL.
- Platform-specific kill: `os.killpg` on macOS, `taskkill /T /F` on Windows — handled by `platform/` module.

### Checkpoint / Recovery (Local State Store)

- SQLite database in user's app data directory.
- Write checkpoint after each step: session state, message thread cursor, current task/step IDs.
- Use WAL mode for concurrent read/write safety.
- On clean session end: delete checkpoint file.
- On crash recovery: load checkpoint, call Session Service `/resume`, reconstruct message thread, continue from last step.

### Testing

- **Pytest markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.platform`
- **Agent loop tests**: Mock LLM client + mock ToolRouter → test step progression, checkpointing, error recovery
- **Tool tests**: Real filesystem operations in temp directories, real subprocess execution with simple commands
- **Policy enforcer tests**: Various policy bundles × tool requests → verify allow/deny/approval-required
- **Platform tests**: `@pytest.mark.skipif(sys.platform != 'darwin')` for macOS-specific, similar for Windows
- **Fixtures**: Pre-built policy bundles, tool requests, LLM responses in `tests/fixtures/`
