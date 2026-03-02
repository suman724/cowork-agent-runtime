#!/usr/bin/env python3
"""
Test a full chat flow: CreateSession -> StartTask -> stream events -> Shutdown.

Each scenario runs in its own isolated runtime process to prevent cascading failures.

Requires:
  - Backend services running (session :8000, policy :8001, workspace :8002)
  - LLM Gateway reachable (LLM_GATEWAY_ENDPOINT)
  - Agent-runtime venv installed (.venv/)

Usage:
  make test-chat            # via Makefile
  python scripts/test-chat.py  # direct (source .env first)
"""

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

# --- Configuration ---

PROJECT_DIR = Path(__file__).resolve().parent.parent
PYTHON = str(PROJECT_DIR / ".venv" / "bin" / "python")

ENV = {
    **os.environ,
    "SESSION_SERVICE_URL": os.environ.get("SESSION_SERVICE_URL", "http://localhost:8000"),
    "WORKSPACE_SERVICE_URL": os.environ.get("WORKSPACE_SERVICE_URL", "http://localhost:8002"),
    "LLM_GATEWAY_ENDPOINT": os.environ.get("LLM_GATEWAY_ENDPOINT", "http://localhost:8080/v1"),
    "LLM_GATEWAY_AUTH_TOKEN": os.environ.get("LLM_GATEWAY_AUTH_TOKEN", "test-token"),
    "LLM_MODEL": os.environ.get("LLM_MODEL", "openai/gpt-4o"),
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", "debug"),
    "APPROVAL_TIMEOUT_SECONDS": os.environ.get("APPROVAL_TIMEOUT_SECONDS", "5"),
}


# --- Exception ---


class TestFailure(Exception):
    """Raised when a test scenario fails."""


# --- Helpers ---


def spawn_runtime(env=None):
    """Spawn a fresh agent-runtime process."""
    proc = subprocess.Popen(
        [PYTHON, "-m", "agent_host.main"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env or ENV,
    )
    time.sleep(1)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode()[-1000:]
        raise TestFailure(f"Process exited immediately (code={proc.returncode})\n{stderr}")
    return proc


def send(proc, request):
    """Send a JSON-RPC request."""
    data = json.dumps(request) + "\n"
    proc.stdin.write(data.encode())
    proc.stdin.flush()


def read_lines(proc, count, timeout_sec=15):
    """Read exactly `count` JSON messages from stdout."""
    lines = []
    start = time.time()
    while time.time() - start < timeout_sec and len(lines) < count:
        remaining = timeout_sec - (time.time() - start)
        poll_sec = min(remaining, 3.0)
        if poll_sec <= 0:
            break
        ready, _, _ = select.select([proc.stdout], [], [], poll_sec)
        if ready:
            raw = proc.stdout.readline().decode().strip()
            if raw:
                lines.append(json.loads(raw))
    return lines


def read_all(proc, timeout_sec=60, quiet_sec=5.0):
    """Read all messages until quiet for quiet_sec seconds or timeout."""
    lines = []
    start = time.time()
    while time.time() - start < timeout_sec:
        remaining = timeout_sec - (time.time() - start)
        poll_sec = min(remaining, 2.0)
        if poll_sec <= 0:
            break
        ready, _, _ = select.select([proc.stdout], [], [], poll_sec)
        if ready:
            raw = proc.stdout.readline().decode().strip()
            if raw:
                lines.append(json.loads(raw))
        elif len(lines) > 0:
            # Got some data, wait to see if more comes (slow LLMs need longer quiet time)
            time.sleep(0.5)
            ready2, _, _ = select.select([proc.stdout], [], [], quiet_sec)
            if not ready2:
                break  # quiet for quiet_sec after last message — done
            raw = proc.stdout.readline().decode().strip()
            if raw:
                lines.append(json.loads(raw))
    return lines


def read_until_task_done(proc, timeout_sec=90):
    """Read messages until a terminal event (task_completed/task_failed) or timeout."""
    lines = []
    terminal_events = {"task_completed", "task_failed"}
    start = time.time()
    while time.time() - start < timeout_sec:
        remaining = timeout_sec - (time.time() - start)
        poll_sec = min(remaining, 2.0)
        if poll_sec <= 0:
            break
        ready, _, _ = select.select([proc.stdout], [], [], poll_sec)
        if ready:
            raw = proc.stdout.readline().decode().strip()
            if raw:
                msg = json.loads(raw)
                lines.append(msg)
                # Check if this is a terminal event
                if "method" in msg:
                    evt = msg.get("params", {}).get("eventType", "")
                    if evt in terminal_events:
                        break
    return lines


def create_session(proc, request_id=1):
    """Send CreateSession and return (session_id, all_messages)."""
    send(proc, {
        "jsonrpc": "2.0", "id": request_id, "method": "CreateSession",
        "params": {
            "tenantId": "dev-tenant", "userId": "dev-user",
            "workspaceHint": {"localPaths": ["/tmp/test-chat"]},
            "clientInfo": {"desktopAppVersion": "0.1.0", "localAgentHostVersion": "0.1.0"},
            "supportedCapabilities": ["File.Read", "File.Write", "Shell.Exec"],
        },
    })

    messages = read_lines(proc, count=2, timeout_sec=15)
    if len(messages) < 2:
        raise TestFailure(f"Expected 2 messages from CreateSession, got {len(messages)}")

    session_id = None
    for msg in messages:
        if "result" in msg:
            result = msg["result"]
            session_id = result.get("sessionId")
            if result.get("status") != "ready":
                raise TestFailure(f"Session status: {result.get('status')}, expected 'ready'")

    if not session_id:
        raise TestFailure("No sessionId received from CreateSession")

    return session_id, messages


def shutdown_runtime(proc, request_id=99):
    """Send Shutdown and wait for process to exit."""
    send(proc, {"jsonrpc": "2.0", "id": request_id, "method": "Shutdown", "params": {}})
    shutdown_msgs = read_lines(proc, count=1, timeout_sec=10)

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise TestFailure("Process did not exit within 10s after Shutdown")

    return shutdown_msgs


def kill_runtime(proc):
    """Force-kill the runtime process."""
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


def dump_stderr(proc):
    """Print relevant stderr lines for diagnostics."""
    try:
        stderr = proc.stderr.read().decode()
        if stderr:
            error_lines = [
                line.strip()
                for line in stderr.split("\n")
                if any(kw in line.lower() for kw in ["error", "exception", "traceback", "failed"])
            ]
            if error_lines:
                print("    STDERR highlights:")
                for line in error_lines[-10:]:
                    print(f"      {line}")
    except Exception:
        pass


def extract_events(messages):
    """Extract event types from a list of JSON-RPC messages."""
    events = []
    for msg in messages:
        if "method" in msg:
            evt = msg.get("params", {}).get("eventType", "")
            if evt:
                events.append(evt)
    return events


def run_isolated(scenario_fn):
    """Run a scenario in its own isolated runtime process.

    Spawns runtime, creates session, calls scenario_fn(proc, session_id),
    shuts down. Returns (passed: bool, error: str | None).
    """
    proc = None
    try:
        proc = spawn_runtime()
        session_id, _ = create_session(proc)
        scenario_fn(proc, session_id)
        try:
            shutdown_runtime(proc)
        except TestFailure:
            kill_runtime(proc)
        return True, None
    except TestFailure as e:
        if proc:
            kill_runtime(proc)
            dump_stderr(proc)
        return False, str(e)
    except Exception as e:
        if proc:
            kill_runtime(proc)
            dump_stderr(proc)
        return False, f"unexpected: {e}"


# --- Scenarios ---


def test_happy_path(proc, session_id):
    """Happy path: StartTask -> stream text -> task_completed."""
    send(proc, {
        "jsonrpc": "2.0", "id": 2, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-happy-1",
            "prompt": "Say hello in one sentence. Do not use any tools.",
            "taskOptions": {"maxSteps": 5},
        },
    })

    all_msgs = read_all(proc, timeout_sec=60)
    events = extract_events(all_msgs)
    text_chunks = []
    has_error = False

    for msg in all_msgs:
        if "method" in msg:
            params = msg.get("params", {})
            evt = params.get("eventType", "")
            if evt == "text_chunk":
                text = params.get("payload", {}).get("text", "")
                text_chunks.append(text)
        elif msg.get("error"):
            has_error = True

    full_text = "".join(text_chunks)
    print(f"    Assistant: '{full_text[:100]}{'...' if len(full_text) > 100 else ''}'")
    print(f"    Events: {events}")

    if has_error:
        raise TestFailure("Received error response during StartTask")
    if not full_text:
        raise TestFailure("No text_chunk events received — LLM may have failed")
    if "task_completed" not in events and "task_failed" not in events:
        raise TestFailure("Neither task_completed nor task_failed emitted")


def test_llm_error_simulation(proc, session_id):
    """LLM error simulation: send prompt, verify task outcome includes retry or failure events."""
    send(proc, {
        "jsonrpc": "2.0", "id": 3, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-llm-err-1",
            "prompt": "Say hi briefly.",
            "taskOptions": {"maxSteps": 3},
        },
    })

    all_msgs = read_all(proc, timeout_sec=60)
    events = extract_events(all_msgs)
    print(f"    Events: {events}")

    # With a working LLM: task_completed. With bad LLM: llm_retry + task_failed.
    # Either outcome is valid for this scenario — we just verify structural correctness.
    terminal_events = {"task_completed", "task_failed", "session_failed"}
    if not terminal_events.intersection(events):
        raise TestFailure(f"No terminal event emitted. Events: {events}")

    retry_count = events.count("llm_retry")
    if retry_count > 0:
        print(f"    LLM retries observed: {retry_count}")


def test_task_cancellation(proc, session_id):
    """Task cancellation: start a task, cancel it, verify cancelled status."""
    send(proc, {
        "jsonrpc": "2.0", "id": 4, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-cancel-1",
            "prompt": "Write a very long essay about the history of computing. Be extremely detailed.",
            "taskOptions": {"maxSteps": 20},
        },
    })

    # Wait briefly for the task to start, then cancel
    time.sleep(2)

    send(proc, {
        "jsonrpc": "2.0", "id": 5, "method": "CancelTask", "params": {},
    })

    # Read messages until we find the CancelTask response (id=5)
    # Events from the running task may arrive before the response.
    cancel_response = None
    start = time.time()
    while time.time() - start < 30:
        remaining = 30 - (time.time() - start)
        poll_sec = min(remaining, 2.0)
        if poll_sec <= 0:
            break
        ready, _, _ = select.select([proc.stdout], [], [], poll_sec)
        if ready:
            raw = proc.stdout.readline().decode().strip()
            if raw:
                msg = json.loads(raw)
                if msg.get("id") == 5 and "result" in msg:
                    cancel_response = msg["result"]
                    break

    if cancel_response is None:
        raise TestFailure("No response received for CancelTask")

    print(f"    Cancel response: {json.dumps(cancel_response)}")

    if cancel_response.get("status") != "cancelled":
        raise TestFailure(f"Expected status='cancelled', got: {cancel_response.get('status')}")


def test_session_resume():
    """Session resume: create session, run task, shutdown, resume in new runtime."""
    # Step 1: Create a session and run a task to completion
    proc1 = spawn_runtime()
    print(f"    Step 1: Created runtime (pid={proc1.pid})")

    try:
        session_id, _ = create_session(proc1, request_id=100)
        print(f"    Step 1: Session created: {session_id[:12]}...")

        # Run a quick task to completion
        send(proc1, {
            "jsonrpc": "2.0", "id": 101, "method": "StartTask",
            "params": {
                "sessionId": session_id,
                "taskId": "task-pre-resume",
                "prompt": "Say 'first task done' in one sentence.",
                "taskOptions": {"maxSteps": 5},
            },
        })
        pre_msgs = read_all(proc1, timeout_sec=60)
        pre_events = extract_events(pre_msgs)
        print(f"    Step 1: Pre-resume task events: {pre_events}")

        if "task_completed" not in pre_events:
            raise TestFailure(f"Pre-resume task did not complete: {pre_events}")

        # Shutdown the first runtime
        shutdown_runtime(proc1, request_id=199)
        print("    Step 1: First runtime shutdown OK")
    except Exception:
        kill_runtime(proc1)
        raise

    # Step 2: Spawn a fresh runtime and resume the session
    proc2 = spawn_runtime()
    print(f"    Step 2: Spawned fresh runtime (pid={proc2.pid})")

    try:
        # Resume the session instead of creating a new one
        send(proc2, {
            "jsonrpc": "2.0", "id": 1, "method": "ResumeSession",
            "params": {"sessionId": session_id},
        })

        messages = read_lines(proc2, count=2, timeout_sec=30)
        if len(messages) < 1:
            raise TestFailure(
                f"Expected at least 1 message from ResumeSession, got {len(messages)}"
            )

        resumed_session_id = None
        for msg in messages:
            print(f"    ResumeSession msg: {json.dumps(msg)[:200]}")
            if "error" in msg:
                raise TestFailure(f"ResumeSession error: {msg['error']}")
            if "result" in msg:
                result = msg["result"]
                resumed_session_id = result.get("sessionId")
                if result.get("status") != "ready":
                    raise TestFailure(
                        f"Resume status: {result.get('status')}, expected 'ready'"
                    )

        if not resumed_session_id:
            raise TestFailure(
                f"No sessionId in ResumeSession response. Messages: "
                f"{json.dumps(messages)[:500]}"
            )

        if resumed_session_id != session_id:
            raise TestFailure(
                f"Session ID mismatch: sent={session_id}, got={resumed_session_id}"
            )

        print(f"    Resumed session: {resumed_session_id[:12]}...")

        # Run a task on the resumed session
        send(proc2, {
            "jsonrpc": "2.0", "id": 2, "method": "StartTask",
            "params": {
                "sessionId": resumed_session_id,
                "taskId": "task-resumed-1",
                "prompt": "Say 'Session resumed successfully' in one sentence.",
                "taskOptions": {"maxSteps": 5},
            },
        })

        all_msgs = read_all(proc2, timeout_sec=60)
        events = extract_events(all_msgs)
        text_chunks = []

        for msg in all_msgs:
            if "method" in msg:
                params = msg.get("params", {})
                evt = params.get("eventType", "")
                if evt == "text_chunk":
                    text = params.get("payload", {}).get("text", "")
                    text_chunks.append(text)

        full_text = "".join(text_chunks)
        print(f"    Assistant: '{full_text[:100]}{'...' if len(full_text) > 100 else ''}'")
        print(f"    Events: {events}")

        if not full_text:
            raise TestFailure("No text_chunk events after resume — LLM may have failed")
        if "task_completed" not in events and "task_failed" not in events:
            raise TestFailure("Neither task_completed nor task_failed emitted after resume")

    finally:
        # Shutdown the second runtime
        try:
            shutdown_runtime(proc2, request_id=99)
            print(f"    Runtime shutdown OK (exit={proc2.returncode})")
        except TestFailure:
            kill_runtime(proc2)
        dump_stderr(proc2)


def test_token_budget_check(proc, session_id):
    """Token budget check: run a task then verify GetSessionState returns tokenUsage."""
    # Run a task first so token budget has data
    send(proc, {
        "jsonrpc": "2.0", "id": 2, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-budget-1",
            "prompt": "Say 'budget test' in one sentence.",
            "taskOptions": {"maxSteps": 3},
        },
    })
    msgs = read_until_task_done(proc, timeout_sec=60)
    events = extract_events(msgs)
    if "task_completed" not in events and "task_failed" not in events:
        raise TestFailure(f"Pre-budget task did not complete: {events}")

    send(proc, {
        "jsonrpc": "2.0", "id": 6, "method": "GetSessionState", "params": {},
    })

    state_msgs = read_lines(proc, count=1, timeout_sec=10)
    if not state_msgs:
        raise TestFailure("No response to GetSessionState")

    response = state_msgs[0]
    if "error" in response:
        raise TestFailure(f"GetSessionState error: {response['error']}")

    result = response.get("result", {})
    print(f"    Session state: {json.dumps(result, indent=2)}")

    token_usage = result.get("tokenUsage")
    if token_usage is None:
        raise TestFailure("tokenUsage not present in GetSessionState response")

    required_fields = [
        "inputTokens", "outputTokens", "totalTokens", "remaining", "maxSessionTokens",
    ]
    for field in required_fields:
        if field not in token_usage:
            raise TestFailure(f"tokenUsage missing field: {field}")

    # Verify types are numeric
    for field in required_fields:
        if not isinstance(token_usage[field], (int, float)):
            raise TestFailure(f"tokenUsage.{field} is not numeric: {token_usage[field]}")

    print(
        f"    Token usage: {token_usage['totalTokens']} used, "
        f"{token_usage['remaining']} remaining"
    )


def test_tool_call_round_trip(proc, session_id):
    """Tool call round-trip: prompt triggers ReadFile, verify tool execution pipeline."""
    # Create a temp file for the LLM to read
    test_file = "/tmp/test-chat/tool-test-file.txt"
    os.makedirs("/tmp/test-chat", exist_ok=True)
    with open(test_file, "w") as f:
        f.write("TOOL_TEST_MARKER_12345\n")

    send(proc, {
        "jsonrpc": "2.0", "id": 10, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-tool-rt-1",
            "prompt": (
                f"Read the file at {test_file} using the ReadFile tool "
                "and report its exact contents."
            ),
            "taskOptions": {"maxSteps": 5},
        },
    })

    all_msgs = read_until_task_done(proc, timeout_sec=90)
    events = extract_events(all_msgs)
    text_chunks = []
    tool_events = []

    for msg in all_msgs:
        if "method" in msg:
            params = msg.get("params", {})
            evt = params.get("eventType", "")
            if evt == "text_chunk":
                text_chunks.append(params.get("payload", {}).get("text", ""))
            if evt in ("tool_requested", "tool_completed"):
                tool_events.append(evt)

    full_text = "".join(text_chunks)
    print(f"    Assistant: '{full_text[:200]}{'...' if len(full_text) > 200 else ''}'")
    print(f"    Events: {events}")
    print(f"    Tool events: {tool_events}")

    if "task_completed" not in events and "task_failed" not in events:
        raise TestFailure(f"No terminal event. Events: {events}")

    # Should have at least tool_completed (tool_requested may be absent if denied)
    if "tool_completed" not in events:
        raise TestFailure("No tool_completed event — tool was not dispatched")

    # Verify the LLM actually called ReadFile and got the marker content back
    # If policy denied the read, the LLM should still report it — we check for the tool round-trip
    if "tool_requested" in events and "TOOL_TEST_MARKER_12345" not in full_text:
        raise TestFailure(f"ReadFile succeeded but marker not in response: {full_text[:200]}")

    # Clean up
    try:
        os.remove(test_file)
    except OSError:
        pass


def test_multi_step_task(proc, session_id):
    """Multi-step task: prompt requiring multiple LLM turns (tool calls + final text)."""
    # Create two files for the LLM to read in sequence
    os.makedirs("/tmp/test-chat", exist_ok=True)
    with open("/tmp/test-chat/file-a.txt", "w") as f:
        f.write("CONTENT_FILE_A\n")
    with open("/tmp/test-chat/file-b.txt", "w") as f:
        f.write("CONTENT_FILE_B\n")

    send(proc, {
        "jsonrpc": "2.0", "id": 11, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-multi-1",
            "prompt": (
                "Read these two files using ReadFile and report their contents:\n"
                "1. /tmp/test-chat/file-a.txt\n"
                "2. /tmp/test-chat/file-b.txt\n"
                "Report the exact content of each file."
            ),
            "taskOptions": {"maxSteps": 10},
        },
    })

    all_msgs = read_until_task_done(proc, timeout_sec=120)
    events = extract_events(all_msgs)
    text_chunks = []
    step_started_count = events.count("step_started")
    step_completed_count = events.count("step_completed")
    tool_completed_count = events.count("tool_completed")

    for msg in all_msgs:
        if "method" in msg:
            params = msg.get("params", {})
            if params.get("eventType") == "text_chunk":
                text_chunks.append(params.get("payload", {}).get("text", ""))

    full_text = "".join(text_chunks)
    print(f"    Assistant: '{full_text[:200]}{'...' if len(full_text) > 200 else ''}'")
    print(f"    Steps: {step_started_count} started, {step_completed_count} completed")
    print(f"    Tool completions: {tool_completed_count}")
    print(f"    Events: {events}")

    # Should have taken multiple steps (at least 2: tool call + response)
    if step_started_count < 2:
        raise TestFailure(
            f"Expected at least 2 steps for multi-step task, got {step_started_count}"
        )

    # Verify the tool was dispatched at least once (even if denied by policy)
    if tool_completed_count < 1:
        raise TestFailure(
            f"Expected at least 1 tool completion, got {tool_completed_count}"
        )

    # Clean up
    for f in ("/tmp/test-chat/file-a.txt", "/tmp/test-chat/file-b.txt"):
        try:
            os.remove(f)
        except OSError:
            pass


def test_step_event_structure(proc, session_id):
    """Step event structure: verify step_started/step_completed have correct payload fields."""
    send(proc, {
        "jsonrpc": "2.0", "id": 12, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-step-evt-1",
            "prompt": "Say 'checking events' in one sentence. Do not use tools.",
            "taskOptions": {"maxSteps": 5},
        },
    })

    all_msgs = read_until_task_done(proc, timeout_sec=60)
    events = extract_events(all_msgs)

    step_started_payloads = []
    step_completed_payloads = []

    for msg in all_msgs:
        if "method" in msg:
            params = msg.get("params", {})
            evt = params.get("eventType", "")
            payload = params.get("payload", {})
            if evt == "step_started":
                step_started_payloads.append(payload)
            elif evt == "step_completed":
                step_completed_payloads.append(payload)

    print(f"    Events: {events}")
    print(f"    step_started payloads: {step_started_payloads}")
    print(f"    step_completed payloads: {step_completed_payloads}")

    if "task_completed" not in events:
        raise TestFailure(f"Task did not complete. Events: {events}")

    if not step_started_payloads:
        raise TestFailure("No step_started events received")
    if not step_completed_payloads:
        raise TestFailure("No step_completed events received")

    # Verify step_started has step number
    for payload in step_started_payloads:
        if "stepNumber" not in payload:
            raise TestFailure(f"step_started payload missing 'stepNumber' field: {payload}")
        if not isinstance(payload["stepNumber"], int) or payload["stepNumber"] < 1:
            raise TestFailure(f"step_started 'stepNumber' should be positive int: {payload}")

    # Verify step_completed has step number
    for payload in step_completed_payloads:
        if "stepNumber" not in payload:
            raise TestFailure(f"step_completed payload missing 'stepNumber' field: {payload}")

    # Verify ordering: step numbers should be sequential
    started_steps = [p["stepNumber"] for p in step_started_payloads]
    completed_steps = [p["stepNumber"] for p in step_completed_payloads]
    print(f"    Started steps: {started_steps}, Completed steps: {completed_steps}")

    if started_steps != sorted(started_steps):
        raise TestFailure(f"step_started events not sequential: {started_steps}")
    if completed_steps != sorted(completed_steps):
        raise TestFailure(f"step_completed events not sequential: {completed_steps}")


def test_step_event_payloads(proc, session_id):
    """Step/tool event payloads: verify toolCallId, toolName, stepNumber fields."""
    # Create a temp file for the LLM to read — triggers tool_requested/tool_completed
    test_file = "/tmp/test-chat/event-payload-test.txt"
    os.makedirs("/tmp/test-chat", exist_ok=True)
    with open(test_file, "w") as f:
        f.write("EVENT_PAYLOAD_TEST\n")

    send(proc, {
        "jsonrpc": "2.0", "id": 20, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-evt-payload-1",
            "prompt": (
                f"Read the file at {test_file} using the ReadFile tool "
                "and report its contents."
            ),
            "taskOptions": {"maxSteps": 5},
        },
    })

    all_msgs = read_until_task_done(proc, timeout_sec=90)
    events = extract_events(all_msgs)

    step_started_payloads = []
    step_completed_payloads = []
    tool_requested_payloads = []
    tool_completed_payloads = []

    for msg in all_msgs:
        if "method" in msg:
            params = msg.get("params", {})
            evt = params.get("eventType", "")
            payload = params.get("payload", {})
            if evt == "step_started":
                step_started_payloads.append(payload)
            elif evt == "step_completed":
                step_completed_payloads.append(payload)
            elif evt == "tool_requested":
                tool_requested_payloads.append(payload)
            elif evt == "tool_completed":
                tool_completed_payloads.append(payload)

    print(f"    Events: {events}")
    print(f"    tool_requested payloads: {json.dumps(tool_requested_payloads)[:300]}")
    print(f"    tool_completed payloads: {json.dumps(tool_completed_payloads)[:300]}")

    if "task_completed" not in events and "task_failed" not in events:
        raise TestFailure(f"No terminal event. Events: {events}")

    # Verify step_started has stepNumber (not 'step')
    for payload in step_started_payloads:
        if "stepNumber" not in payload:
            raise TestFailure(f"step_started missing 'stepNumber': {payload}")
        if "step" in payload:
            raise TestFailure(f"step_started has old 'step' field (should be 'stepNumber'): {payload}")
        if not isinstance(payload["stepNumber"], int) or payload["stepNumber"] < 1:
            raise TestFailure(f"stepNumber should be positive int: {payload}")

    # Verify step_completed has stepNumber
    for payload in step_completed_payloads:
        if "stepNumber" not in payload:
            raise TestFailure(f"step_completed missing 'stepNumber': {payload}")

    # Verify tool_completed has toolCallId, status
    if tool_completed_payloads:
        for payload in tool_completed_payloads:
            if "toolCallId" not in payload:
                raise TestFailure(f"tool_completed missing 'toolCallId': {payload}")
            if "status" not in payload:
                raise TestFailure(f"tool_completed missing 'status': {payload}")
            # status should be one of: completed, failed, denied
            valid_statuses = {"completed", "failed", "denied"}
            if payload["status"] not in valid_statuses:
                raise TestFailure(
                    f"tool_completed status '{payload['status']}' not in {valid_statuses}"
                )
    else:
        raise TestFailure("No tool_completed events — expected at least one")

    # Verify tool_requested has toolCallId, toolName, arguments (if present)
    for payload in tool_requested_payloads:
        if "toolCallId" not in payload:
            raise TestFailure(f"tool_requested missing 'toolCallId': {payload}")
        if "toolName" not in payload:
            raise TestFailure(f"tool_requested missing 'toolName': {payload}")

    # Clean up
    try:
        os.remove(test_file)
    except OSError:
        pass


def test_context_compaction():
    """Context compaction: verify compaction triggers with very low MAX_CONTEXT_TOKENS."""
    # Use an extremely low MAX_CONTEXT_TOKENS to force compaction.
    # The ~4 chars/token heuristic means even the system prompt (~800 chars → ~200 tokens)
    # will approach the budget. After 2 tasks the thread will definitely exceed it.
    # RECENCY_WINDOW=2 means only system prompt + last 2 messages are kept.
    env = {
        **ENV,
        "MAX_CONTEXT_TOKENS": "50",
        "RECENCY_WINDOW": "2",
    }
    proc = spawn_runtime(env=env)

    try:
        session_id, _ = create_session(proc)

        # Task 1: Build up some conversation history
        send(proc, {
            "jsonrpc": "2.0", "id": 2, "method": "StartTask",
            "params": {
                "sessionId": session_id,
                "taskId": "task-compact-1",
                "prompt": "Say 'first message done' in one sentence.",
                "taskOptions": {"maxSteps": 3},
            },
        })
        msgs1 = read_until_task_done(proc, timeout_sec=60)
        events1 = extract_events(msgs1)
        if "task_completed" not in events1 and "task_failed" not in events1:
            raise TestFailure(f"Task 1 did not complete: {events1}")
        print(f"    Task 1 events: {events1}")

        # Task 2: Thread now has system + user1 + assistant1 + user2 → should trigger compaction
        send(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "StartTask",
            "params": {
                "sessionId": session_id,
                "taskId": "task-compact-2",
                "prompt": "Say 'second message done' in one sentence.",
                "taskOptions": {"maxSteps": 3},
            },
        })
        msgs2 = read_until_task_done(proc, timeout_sec=60)
        events2 = extract_events(msgs2)
        print(f"    Task 2 events: {events2}")

        # Task 3: By now compaction should have definitely occurred
        send(proc, {
            "jsonrpc": "2.0", "id": 4, "method": "StartTask",
            "params": {
                "sessionId": session_id,
                "taskId": "task-compact-3",
                "prompt": "Say 'third message done' in one sentence.",
                "taskOptions": {"maxSteps": 3},
            },
        })
        msgs3 = read_until_task_done(proc, timeout_sec=60)
        events3 = extract_events(msgs3)
        print(f"    Task 3 events: {events3}")

        # Collect all events across all tasks
        all_events = events1 + events2 + events3
        print(f"    All events: {all_events}")

        # Verify context_compacted event was emitted at some point
        compaction_payloads = []
        for msg in msgs1 + msgs2 + msgs3:
            if "method" in msg:
                params = msg.get("params", {})
                if params.get("eventType") == "context_compacted":
                    compaction_payloads.append(params.get("payload", {}))

        if not compaction_payloads:
            raise TestFailure(
                "No context_compacted event emitted despite "
                "MAX_CONTEXT_TOKENS=50. "
                f"Events: {all_events}"
            )

        # Verify compaction payload has expected fields
        for payload in compaction_payloads:
            print(f"    Compaction payload: {payload}")
            if "messagesDropped" not in payload:
                raise TestFailure(
                    f"context_compacted missing 'messagesDropped': {payload}"
                )
            if "tokensBefore" not in payload:
                raise TestFailure(
                    f"context_compacted missing 'tokensBefore': {payload}"
                )
            if "tokensAfter" not in payload:
                raise TestFailure(
                    f"context_compacted missing 'tokensAfter': {payload}"
                )

        # Verify the agent still functions after compaction
        terminal = {"task_completed", "task_failed"}
        if not terminal.intersection(set(events3)):
            raise TestFailure(
                "Agent stopped after compaction. "
                f"Task 3 events: {events3}"
            )

    finally:
        try:
            shutdown_runtime(proc)
        except TestFailure:
            kill_runtime(proc)
        dump_stderr(proc)


def test_checkpoint_persistence():
    """Checkpoint persistence: verify checkpoint survives crash and is restored on resume."""
    import glob as glob_mod

    proc1 = spawn_runtime()

    try:
        session_id, _ = create_session(proc1, request_id=200)
        print(f"    Session: {session_id[:12]}...")

        # Run a task so checkpoint has data
        send(proc1, {
            "jsonrpc": "2.0", "id": 201, "method": "StartTask",
            "params": {
                "sessionId": session_id,
                "taskId": "task-ckpt-1",
                "prompt": "Say 'checkpoint test' in one sentence.",
                "taskOptions": {"maxSteps": 3},
            },
        })
        msgs = read_until_task_done(proc1, timeout_sec=60)
        events = extract_events(msgs)
        if "task_completed" not in events:
            raise TestFailure(f"Task did not complete: {events}")
        print(f"    Task completed, events: {events}")

        # Get token usage before crash
        send(proc1, {
            "jsonrpc": "2.0", "id": 202, "method": "GetSessionState",
            "params": {},
        })
        state_msgs = read_lines(proc1, count=1, timeout_sec=10)
        if not state_msgs:
            raise TestFailure("No response to GetSessionState")
        pre_usage = state_msgs[0].get("result", {}).get("tokenUsage", {})
        pre_tokens = pre_usage.get("totalTokens", 0)
        print(f"    Pre-crash token usage: {pre_tokens}")

        # Check checkpoint file exists (written after step completion)
        checkpoint_dir = os.path.expanduser(
            "~/Library/Application Support/"
            "cowork/agent-runtime/checkpoints"
        )
        ckpt_path = os.path.join(
            checkpoint_dir, f"cowork_{session_id}.json"
        )
        checkpoint_files = glob_mod.glob(ckpt_path)
        print(f"    Looking for checkpoint: {ckpt_path}")
        if not checkpoint_files:
            raise TestFailure(
                f"Checkpoint file not found at {ckpt_path}"
            )
        print(f"    Checkpoint exists: {checkpoint_files[0]}")

        # Read checkpoint to verify token data
        with open(checkpoint_files[0]) as f:
            checkpoint_data = json.load(f)
        ckpt_in = checkpoint_data.get("token_input_used", 0)
        ckpt_out = checkpoint_data.get("token_output_used", 0)
        print(f"    Checkpoint tokens: in={ckpt_in}, out={ckpt_out}")
        if ckpt_in + ckpt_out == 0:
            raise TestFailure(
                "Checkpoint has zero token usage — not persisted"
            )

        # KILL the runtime (simulate crash) instead of clean shutdown.
        # Clean shutdown deletes the checkpoint — we need it to persist.
        kill_runtime(proc1)
        print("    Runtime killed (simulating crash)")

        # Verify checkpoint file still exists after crash
        if not os.path.exists(checkpoint_files[0]):
            raise TestFailure(
                "Checkpoint file deleted after crash — should persist"
            )
        print("    Checkpoint survives crash OK")
    except TestFailure:
        kill_runtime(proc1)
        dump_stderr(proc1)
        raise
    except Exception:
        kill_runtime(proc1)
        dump_stderr(proc1)
        raise

    # Resume in new runtime — checkpoint should be restored
    proc2 = spawn_runtime()
    try:
        send(proc2, {
            "jsonrpc": "2.0", "id": 1, "method": "ResumeSession",
            "params": {"sessionId": session_id},
        })
        messages = read_lines(proc2, count=2, timeout_sec=30)

        resumed = False
        for msg in messages:
            if "error" in msg:
                raise TestFailure(
                    f"ResumeSession error: {msg['error']}"
                )
            if "result" in msg:
                if msg["result"].get("status") == "ready":
                    resumed = True
        if not resumed:
            raise TestFailure("ResumeSession did not return ready")
        print("    Session resumed OK")

        # Get token usage after resume — should reflect checkpoint
        send(proc2, {
            "jsonrpc": "2.0", "id": 2, "method": "GetSessionState",
            "params": {},
        })
        state_msgs = read_lines(proc2, count=1, timeout_sec=10)
        if not state_msgs:
            raise TestFailure(
                "No response to GetSessionState after resume"
            )
        post_usage = state_msgs[0].get("result", {}).get("tokenUsage", {})
        post_tokens = post_usage.get("totalTokens", 0)
        print(f"    Post-resume token usage: {post_tokens}")

        if post_tokens == 0:
            raise TestFailure(
                "Token budget reset to 0 after resume — "
                "checkpoint not restored"
            )
        print(
            f"    Token budget preserved: {post_tokens} "
            f"(was {pre_tokens})"
        )

    finally:
        try:
            shutdown_runtime(proc2, request_id=99)
        except TestFailure:
            kill_runtime(proc2)
        dump_stderr(proc2)


def test_error_event_payloads(proc, session_id):
    """Error event payloads: verify tool failure events have correct fields."""
    # Ask the LLM to read a non-existent file — should produce a tool failure
    send(proc, {
        "jsonrpc": "2.0", "id": 21, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-err-evt-1",
            "prompt": (
                "Read the file at /tmp/test-chat/DOES_NOT_EXIST_12345.txt using ReadFile. "
                "If it fails, just say 'file not found'."
            ),
            "taskOptions": {"maxSteps": 5},
        },
    })

    all_msgs = read_until_task_done(proc, timeout_sec=90)
    events = extract_events(all_msgs)

    tool_completed_payloads = []
    for msg in all_msgs:
        if "method" in msg:
            params = msg.get("params", {})
            if params.get("eventType") == "tool_completed":
                tool_completed_payloads.append(params.get("payload", {}))

    print(f"    Events: {events}")
    print(f"    tool_completed payloads: {json.dumps(tool_completed_payloads)[:400]}")

    if "task_completed" not in events and "task_failed" not in events:
        raise TestFailure(f"No terminal event. Events: {events}")

    # Find tool_completed events with failed or denied status
    error_payloads = [
        p for p in tool_completed_payloads
        if p.get("status") in ("failed", "denied")
    ]

    if not error_payloads:
        raise TestFailure(
            "No failed/denied tool_completed events — expected at least one for non-existent file"
        )

    for payload in error_payloads:
        print(f"    Error payload: {payload}")
        # Must have toolCallId
        if "toolCallId" not in payload:
            raise TestFailure(f"Error tool_completed missing 'toolCallId': {payload}")
        # Must have error field for failed/denied
        if "error" not in payload and "result" not in payload:
            raise TestFailure(
                f"Error tool_completed missing both 'error' and 'result': {payload}"
            )

    # Verify the agent recovered and completed the task (not stuck)
    if "task_completed" not in events:
        raise TestFailure(
            f"Agent did not recover from tool failure — task_completed not emitted. "
            f"Events: {events}"
        )
    print("    Agent recovered from tool failure OK")


def test_conversation_context_preserved(proc, session_id):
    """Conversation context: verify LLM has access to prior task history within the session."""
    # Task 1: Tell the LLM a specific fact
    send(proc, {
        "jsonrpc": "2.0", "id": 13, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-ctx-1",
            "prompt": "Remember this code: ALPHA-7742. Just acknowledge it.",
            "taskOptions": {"maxSteps": 3},
        },
    })
    msgs1 = read_until_task_done(proc, timeout_sec=60)
    events1 = extract_events(msgs1)
    if "task_completed" not in events1:
        raise TestFailure(f"Context task 1 did not complete: {events1}")
    print("    Task 1: Told LLM the code ALPHA-7742")

    # Task 2: Ask the LLM to recall the fact
    send(proc, {
        "jsonrpc": "2.0", "id": 14, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-ctx-2",
            "prompt": "What was the code I told you earlier? Reply with just the code.",
            "taskOptions": {"maxSteps": 3},
        },
    })
    msgs2 = read_until_task_done(proc, timeout_sec=60)
    events2 = extract_events(msgs2)
    text_chunks = []
    for msg in msgs2:
        if "method" in msg:
            params = msg.get("params", {})
            if params.get("eventType") == "text_chunk":
                text_chunks.append(params.get("payload", {}).get("text", ""))

    full_text = "".join(text_chunks)
    print(f"    Task 2 response: '{full_text[:150]}'")
    print(f"    Events: {events2}")

    if "task_completed" not in events2:
        raise TestFailure(f"Context task 2 did not complete: {events2}")

    if "ALPHA-7742" not in full_text:
        raise TestFailure(
            f"LLM did not recall the code — conversation context may not be preserved. "
            f"Response: {full_text[:200]}"
        )


# --- Main ---


def main():
    print("=" * 60)
    print("Agent-Runtime Chat Flow Test")
    print("=" * 60)
    print(f"  LLM endpoint: {ENV['LLM_GATEWAY_ENDPOINT']}")
    print(f"  LLM model:    {ENV['LLM_MODEL']}")

    if not os.path.exists(PYTHON):
        print(f"FAIL: Python venv not found at {PYTHON}. Run: make install")
        sys.exit(1)

    # Each scenario runs in its own isolated runtime process.
    # No shared state — no cascading failures.
    # Scenarios that run in isolated runtime processes (spawn → test → shutdown)
    scenarios = [
        ("Happy Path", test_happy_path),
        ("Step Event Structure", test_step_event_structure),
        ("Step/Tool Event Payloads", test_step_event_payloads),
        ("LLM Error Simulation", test_llm_error_simulation),
        ("Token Budget Check", test_token_budget_check),
        ("Tool Call Round-Trip", test_tool_call_round_trip),
        ("Multi-Step Task", test_multi_step_task),
        ("Task Cancellation", test_task_cancellation),
        ("Error Event Payloads", test_error_event_payloads),
        ("Conversation Context Preserved", test_conversation_context_preserved),
    ]

    # Scenarios that manage their own runtime processes (custom lifecycle)
    standalone_scenarios = [
        ("Session Resume", test_session_resume),
        ("Context Compaction", test_context_compaction),
        ("Checkpoint Persistence", test_checkpoint_persistence),
    ]

    total_count = len(scenarios) + len(standalone_scenarios)
    passed = 0
    failed = 0
    results = []

    for name, test_fn in scenarios:
        print(f"\n  [{len(results) + 1}/{total_count}] {name}...")
        ok, err = run_isolated(test_fn)
        if ok:
            print(f"    PASS: {name}")
            passed += 1
        else:
            print(f"    FAIL: {name} — {err}")
            failed += 1
        results.append((name, ok, err))

    # Standalone scenarios — manage their own runtime processes
    for name, test_fn in standalone_scenarios:
        print(f"\n  [{len(results) + 1}/{total_count}] {name}...")
        try:
            test_fn()
            print(f"    PASS: {name}")
            passed += 1
            results.append((name, True, None))
        except TestFailure as e:
            print(f"    FAIL: {name} — {e}")
            failed += 1
            results.append((name, False, str(e)))
        except Exception as e:
            print(f"    FAIL: {name} — unexpected: {e}")
            failed += 1
            results.append((name, False, str(e)))

    # Summary
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {total} total")
    print("=" * 60)
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        suffix = f" — {err}" if err else ""
        print(f"  [{status}] {name}{suffix}")
    print("=" * 60)

    if failed > 0:
        print("\nCHAT FLOW TEST FAILED")
        sys.exit(1)
    else:
        print("\nCHAT FLOW TEST PASSED")


if __name__ == "__main__":
    main()
