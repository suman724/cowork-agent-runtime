# cowork-agent-runtime

Local Agent Host and Tool Runtime for the cowork desktop agent system. Runs as a child process of the Desktop App, executing tool calls (file operations, shell commands, HTTP requests) on the local machine.

## Setup

```bash
# Install dependencies (requires cowork-platform sibling repo)
make install

# Run CI checks
make check
```

## Development

```bash
make lint          # Run ruff linter
make format        # Auto-format code
make typecheck     # Run mypy strict mode
make test          # Run unit tests
make coverage      # Run tests with coverage report
```

## Architecture

Two packages with a strict boundary (no cross-imports):

- **`tool_runtime/`** — Tool router, built-in tools (file, shell, network), platform adapters, output formatting
- **`agent_host/`** — Agent loop, JSON-RPC server, session/LLM clients, policy enforcer (Phase 1C.2)

Communication between packages is via the `ToolRouter` interface only.

## Built-in Tools

| Tool | Capability | Description |
|------|-----------|-------------|
| `ReadFile` | `File.Read` | Read file contents with encoding detection |
| `WriteFile` | `File.Write` | Atomic file write with diff generation |
| `DeleteFile` | `File.Delete` | Delete files (not directories) |
| `RunCommand` | `Shell.Exec` | Execute shell commands with timeout |
| `HttpRequest` | `Network.Http` | HTTP requests with SSRF prevention |
