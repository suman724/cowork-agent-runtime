# cowork-agent-runtime

Local Agent Host and Tool Runtime for the Cowork agent system. Supports two transport modes:

- **stdio** (default): Spawned by the Desktop App as a child process, communicates via JSON-RPC 2.0 over stdin/stdout.
- **http**: Runs as an HTTP/SSE server for web/sandbox mode. Exposes JSON-RPC via `POST /rpc`, events via `GET /events` (SSE), and file operations via `/upload` and `/files`.

## Transport Modes

### stdio mode (Desktop App)

```bash
make run          # Start in stdio mode (default)
```

The Desktop App spawns the agent-runtime as a child process. JSON-RPC 2.0 requests are sent on stdin, responses and event notifications on stdout.

### HTTP mode (Web / Sandbox)

```bash
make run-sandbox  # Start in HTTP mode on localhost:8080
```

Or with explicit options:

```bash
python -m agent_host.main --transport http --host 0.0.0.0 --port 8080 --workspace-dir ./workspace
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rpc` | POST | JSON-RPC 2.0 dispatch (same methods as stdio) |
| `/events` | GET | SSE event stream with replay (`?since={id}`) |
| `/health` | GET | Liveness probe (always 200) |
| `/ready` | GET | Readiness probe (200 when SessionManager initialized) |
| `/upload` | POST | Multipart file upload to workspace directory |
| `/files/{path}` | GET | Download file from workspace |
| `/files` | GET | List workspace files (or zip archive with `?archive=true`) |

## JSON-RPC API

Both transports use the same JSON-RPC 2.0 methods:

| Method | Description |
|--------|-------------|
| `CreateSession` | Handshake with Session Service, initialize agent loop with policy bundle |
| `ResumeSession` | Resume a completed/failed session — re-fetch policy, restore history, reuse session ID |
| `StartTask` | Start an agent work cycle from a user prompt. Accepts `taskOptions.maxSteps` (1-200, default 50) to limit LLM calls per task |
| `CancelTask` | Cancel the currently running task |
| `GetSessionState` | Return session status, active task, token usage, `currentStep`, and `maxSteps` |
| `ApproveAction` | Deliver a user approval/denial for a pending tool call |
| `GetEvents` | Return buffered events since a given ID (for replay after reconnect) |
| `Shutdown` | Cancel task, clean up session, close connections |

In stdio mode, streaming events are sent as JSON-RPC notifications (`SessionEvent`) on stdout. In HTTP mode, events are streamed via SSE on `GET /events`. All events include a monotonic `eventId` for replay tracking.

## Built-in Tools

| Tool | Capability | Description |
|------|-----------|-------------|
| `ReadFile` | `File.Read` | Read file contents with encoding detection |
| `WriteFile` | `File.Write` | Atomic file write with diff generation |
| `EditFile` | `File.Write` | Exact-match find-and-replace editing |
| `MultiEdit` | `File.Write` | Batch multiple find-and-replace edits atomically |
| `DeleteFile` | `File.Delete` | Delete a file |
| `CreateDirectory` | `File.Write` | Create directories without shell commands |
| `MoveFile` | `File.Write` | Move or rename files and directories |
| `ListDirectory` | `File.Read` | List files and directories at a path |
| `FindFiles` | `File.Read` | Glob-pattern file discovery |
| `GrepFiles` | `File.Read` | Regex search across files |
| `ViewImage` | `File.Read` | Read image for multimodal LLM |
| `RunCommand` | `Shell.Exec` | Execute shell commands with timeout |
| `HttpRequest` | `Network.Http` | HTTP requests with SSRF prevention |
| `FetchUrl` | `Network.Http` | Fetch URL, convert HTML to markdown |
| `WebSearch` | `Search.Web` | Web search via Tavily API |
| `ExecuteCode` | `Code.Execute` | Execute Python scripts with output capture |

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

# Run locally
make run           # stdio mode (Desktop App)
make run-sandbox   # HTTP mode on localhost:8080
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--transport` | `stdio` | Transport mode: `stdio` (Desktop App) or `http` (web/sandbox) |
| `--host` | `0.0.0.0` | HTTP server bind address (only with `--transport http`) |
| `--port` | `8080` | HTTP server port (only with `--transport http`) |
| `--workspace-dir` | — | Workspace directory for file upload/download (only with `--transport http`) |

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
| `SKILLS_DIR` | `~/.cowork/skills/` | Override user skills directory |
| `SESSION_ID` | — | Pre-assigned session ID (sandbox mode — triggers self-registration) |
| `REGISTRATION_TOKEN` | — | Token for sandbox self-registration with Session Service |
| `SANDBOX_LOCAL_MODE` | `false` | Skip ECS metadata, use localhost (sandbox local dev) |

## Architecture

Two packages with a strict boundary — no cross-imports, communication via `ToolRouter` interface only.

### agent_host/

Custom agent loop with production-grade harness:

| Module | Purpose |
|--------|---------|
| `server/` | Transport layer (Transport protocol, StdioTransport, HttpTransport), JSON-RPC 2.0 (parse, serialize, dispatch, handlers), EventBuffer (SSE replay) |
| `loop/` | Core agent loop, tool executor, agent-internal tools, error recovery, sub-agents |
| `llm/` | LLM Gateway streaming client (openai SDK), response models, error classifier |
| `thread/` | Message thread management, context compaction, token counting |
| `memory/` | Working memory: task tracker, plan, notes (injected per-turn) |
| `skills/` | Skill definitions, loader (built-in/user/workspace/policy), executor |
| `session/` | Session/Workspace HTTP clients, checkpoint manager, SessionManager |
| `policy/` | Policy enforcer, path/command/domain matchers, risk assessor |
| `budget/` | Session token budget tracking |
| `approval/` | Approval gate (asyncio Futures for user approval flow) |
| `events/` | Event emitter (JSON-RPC notifications + structlog) |
| `sandbox/` | Sandbox mode: self-registration (startup.py), workspace file sync (workspace_sync.py) |

### tool_runtime/

Local tool execution engine:

| Module | Purpose |
|--------|---------|
| `router/` | Tool registry and dispatch |
| `tools/file/` | ReadFile, WriteFile, EditFile, MultiEdit, DeleteFile, CreateDirectory, MoveFile, ListDirectory, FindFiles, GrepFiles, ViewImage |
| `tools/shell/` | RunCommand with platform-specific process management |
| `tools/network/` | HttpRequest, FetchUrl, WebSearch |
| `tools/code/` | ExecuteCode (Python script execution) |
| `code/` | Code execution engine (PythonExecutor) |
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
| `starlette` | ASGI framework for HttpTransport (web/sandbox mode) |
| `uvicorn` | ASGI server for HttpTransport |
| `python-multipart` | Multipart form parsing for file upload |
