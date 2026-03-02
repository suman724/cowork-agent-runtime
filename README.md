# cowork-agent-runtime

Local Agent Host and Tool Runtime for the Cowork desktop agent system. Runs as a child process of the Desktop App, executing the agent loop locally for fast file/tool interactions while routing LLM calls through an external gateway.

## JSON-RPC API

The Desktop App communicates with the Agent Host via JSON-RPC 2.0 over stdio (newline-delimited JSON).

| Method | Description |
|--------|-------------|
| `CreateSession` | Handshake with Session Service, initialize agent loop with policy bundle |
| `ResumeSession` | Resume a completed/failed session — re-fetch policy, restore history, reuse session ID |
| `StartTask` | Start an agent work cycle from a user prompt. Accepts `taskOptions.maxSteps` (1-200, default 50) to limit LLM calls per task |
| `CancelTask` | Cancel the currently running task |
| `GetSessionState` | Return session status, active task, token usage, `currentStep`, and `maxSteps` |
| `ApproveAction` | Deliver a user approval/denial for a pending tool call |
| `Shutdown` | Cancel task, clean up session, close connections |

Streaming events are sent as JSON-RPC notifications (`SessionEvent`) on stdout.

## Built-in Tools

| Tool | Capability | Description |
|------|-----------|-------------|
| `ReadFile` | `File.Read` | Read file contents with encoding detection |
| `WriteFile` | `File.Write` | Atomic file write with diff generation |
| `DeleteFile` | `File.Delete` | Delete files (not directories) |
| `RunCommand` | `Shell.Exec` | Execute shell commands with timeout |
| `HttpRequest` | `Network.Http` | HTTP requests with SSRF prevention |

## Development

```bash
# Install dependencies (requires cowork-platform sibling repo)
make install

# Run CI gate (lint + format-check + typecheck + tests)
make check

# Individual checks
make lint          # Run ruff linter
make format        # Auto-format code
make typecheck     # Run mypy strict mode
make test          # Run unit tests
make coverage      # Run tests with coverage report
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_GATEWAY_ENDPOINT` | — | LLM Gateway URL (required) |
| `LLM_GATEWAY_AUTH_TOKEN` | — | LLM Gateway auth token (required) |
| `SESSION_SERVICE_URL` | — | Session Service URL (required) |
| `WORKSPACE_SERVICE_URL` | — | Workspace Service URL (required) |
| `DEFAULT_MAX_STEPS` | `50` | Default max LLM calls per task (overridden by `taskOptions.maxSteps`) |
| `MAX_CONTEXT_TOKENS` | `100000` | Context window token budget — oldest messages truncated when exceeded |
| `CHECKPOINT_DIR` | Platform app data | Directory for session checkpoint files |
| `APPROVAL_TIMEOUT_SECONDS` | `300` | Timeout for pending approval requests |
| `LOG_LEVEL` | `info` | Structured logging level (debug, info, warning, error) |
| `LLM_MODEL` | `gpt-4o` | LLM model identifier for OpenAI-compatible gateway |

## Architecture

Two packages with a strict boundary — no cross-imports, communication via `ToolRouter` interface only.

### agent_host/

Custom agent loop with production-grade harness:

| Module | Purpose |
|--------|---------|
| `server/` | JSON-RPC 2.0 server (parse, serialize, stdio transport, dispatch, handlers) |
| `loop/` | Core agent loop, tool executor, agent-internal tools, error recovery, sub-agents |
| `llm/` | LLM Gateway streaming client (openai SDK), response models, error classifier |
| `thread/` | Message thread management, context compaction, token counting |
| `memory/` | Working memory: task tracker, plan, notes (injected per-turn) |
| `skills/` | Skill definitions, loader (built-in/YAML/policy), executor |
| `session/` | Session/Workspace HTTP clients, checkpoint manager, SessionManager |
| `policy/` | Policy enforcer, path/command/domain matchers, risk assessor |
| `budget/` | Session token budget tracking |
| `approval/` | Approval gate (asyncio Futures for user approval flow) |
| `events/` | Event emitter (JSON-RPC notifications + structlog) |

### tool_runtime/

Local tool execution engine:

| Module | Purpose |
|--------|---------|
| `router/` | Tool registry and dispatch |
| `tools/file/` | ReadFile, WriteFile, DeleteFile |
| `tools/shell/` | RunCommand with platform-specific process management |
| `tools/network/` | HttpRequest with SSRF prevention |
| `platform/` | OS abstraction for macOS/Windows (path handling, shell resolution) |
| `output/` | Output formatting, truncation, artifact extraction |

## Dependencies

| Library | Purpose |
|---------|---------|
| `openai` | LLM Gateway streaming client (AsyncOpenAI, OpenAI-compatible endpoint) |
| `tenacity` | Retry with exponential backoff for HTTP clients |
| `httpx` | Async HTTP for backend service calls |
| `pydantic` | Data validation (cowork-platform contracts) |
| `structlog` | Structured logging to stderr |
