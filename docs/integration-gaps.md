# Agent Loop Integration Gaps

Tracked gaps between the custom agent loop plan and implementation, plus desktop app integration issues.

## Desktop App Integration Issues

### CRITICAL: Tool Calls Invisible in UI

**`tool_requested` missing `toolCallId` and `arguments`**

Runtime emits:
```json
{"toolName": "ReadFile", "capability": "File.Read"}
```

Desktop expects:
```json
{"toolCallId": "call_abc123", "toolName": "ReadFile", "arguments": {"path": "/foo"}}
```

The desktop guard `if (toolCallId && toolName)` fails — no tool call cards ever appear.

**`tool_completed` missing `toolCallId`, `result`, and `error`**

Runtime emits:
```json
{"toolName": "ReadFile", "status": "succeeded"}
```

Desktop expects:
```json
{"toolCallId": "call_abc123", "status": "succeeded", "result": "file contents...", "error": null}
```

Tool card status never updates, no output shown.

**Fix:** Update `emit_tool_requested` and `emit_tool_completed` in `event_emitter.py` to include `toolCallId`, `arguments`, `result`, `error`. Thread the tool call ID from `ToolExecutor` through to the emitter.

**Status:** DONE

---

### MEDIUM: Step Counter Broken

Runtime sends `{"step": 1}`, desktop reads `payload.stepNumber`. Step counter never updates.

**Fix:** Change runtime to emit `stepNumber` instead of `step` in `step_started` payload.

**Status:** DONE

---

### MEDIUM: Denied Tools Show as Completed

Runtime sends `status: "denied"`, desktop maps anything not `"failed"` to `"completed"`. A policy-denied tool appears as successful.

**Fix:** Desktop-side change — add `"denied"` to `ToolCallInfo.status` type union, map `"denied"` status in `use-session-events.ts`, add `denied` variant/label in `ToolCallCard.tsx`.

**Status:** DONE (desktop-app: types.ts, use-session-events.ts, ToolCallCard.tsx updated)

---

### LOW: Event Schema Drift

`event-envelope.json` and `event-types.json` in `cowork-platform` don't include `task_completed`, `task_failed`, `llm_retry`, `step_started`, `step_completed`, `context_compacted`.

**Fix:** Update `cowork-platform` schemas (enums/event-types.json, schemas/event-envelope.json, jsonrpc/session-event.json) and SDK constants (Python + TypeScript) to include `task_completed`, `task_failed`, `llm_retry`, `context_compacted`.

**Status:** DONE (all three schema files + both SDK constants updated)

---

## Implementation Gaps (Plan vs Code)

### Working Memory Not Injected (Wave 3-4)

Modules exist (`WorkingMemory`, `TaskTracker`, `Plan`) but working memory is never injected into the LLM payload. The agent loop doesn't pass working memory to `MessageThread.build_llm_payload()`. Agent-internal tools (`TaskTracker`, `CreatePlan`) execute but their output is invisible to the LLM on subsequent turns.

**Fix:** Add `working_memory` parameter to `AgentLoop`. Inject `working_memory.render()` as a system message after the static system prompt on every turn.

**Status:** DONE

---

### Error Recovery Not Wired (Wave 5)

`ErrorRecovery` module exists but `AgentLoop.run()` never calls it. No consecutive failure tracking, no loop detection, no reflection prompt injection.

**Fix:** Add `error_recovery` parameter to `AgentLoop`. After tool execution, call `record_tool_failure/success`. Before each LLM call, check `should_inject_reflection()` and `detect_loop()`, inject prompts via `thread.add_system_injection()`.

**Status:** DONE

---

### Sub-Agents Not Callable (Wave 6)

`SubAgentManager` existed but `SpawnAgent` was not in `AgentToolHandler.AGENT_TOOL_NAMES`. The loop didn't instantiate a `SubAgentManager`.

**Fix:** `SubAgentManager` was absorbed into `LoopRuntime.spawn_sub_agent()` as part of the Loop Strategy refactor. `SpawnAgent` is registered in `AgentToolHandler` and delegates to `LoopRuntime`.

**Status:** DONE

---

### Skills Not Loaded or Callable (Wave 7)

`SkillLoader` and `SkillExecutor` existed but `SessionManager` didn't load skills during `create_session()`. Skills were not registered as available tools.

**Fix:** `SkillExecutor` was absorbed into `LoopRuntime.execute_skill()` as part of the Loop Strategy refactor. Skills are loaded in `SessionManager.create_session()` and registered as agent-internal tools in `AgentToolHandler`, which delegates to `LoopRuntime`.

**Status:** DONE

---

### Loop Strategy Refactor (Architecture)

Monolithic `AgentLoop` mixed orchestration, context assembly, and infrastructure. Made it impossible to experiment with alternative strategies.

**Fix:** Decomposed into three layers: `LoopRuntime` (infrastructure primitives), `LoopStrategy` protocol (orchestration), `ReactLoop` (default strategy). `AgentLoop` is now a thin alias for `ReactLoop`. `SubAgentManager` and `SkillExecutor` absorbed into `LoopRuntime`. See `cowork-infra/docs/components/loop-strategy.md`.

**Status:** DONE
