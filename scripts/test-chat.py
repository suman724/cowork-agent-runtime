#!/usr/bin/env python3
"""
Test a full chat flow: CreateSession -> StartTask -> stream events -> Shutdown.

Also runs additional scenarios: LLM error simulation, task cancellation, token budget check.

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


def read_all(proc, timeout_sec=60):
    """Read all messages until quiet for 3 seconds or timeout."""
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
            # Got some data, wait a bit more to see if more comes
            time.sleep(0.5)
            ready2, _, _ = select.select([proc.stdout], [], [], 2.0)
            if not ready2:
                break  # quiet for 2.5s after last message — done
            raw = proc.stdout.readline().decode().strip()
            if raw:
                lines.append(json.loads(raw))
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
                print("  STDERR highlights:")
                for line in error_lines[-10:]:
                    print(f"    {line}")
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


# --- Scenarios ---


def test_happy_path(proc, session_id):
    """Happy path: StartTask -> stream text -> task_completed."""
    print("\n  Scenario: Happy Path")
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

    return all_msgs


def test_llm_error_simulation(proc, session_id):
    """LLM error simulation: send prompt, verify task outcome includes retry or failure events."""
    print("\n  Scenario: LLM Error Simulation")
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

    return all_msgs


def test_task_cancellation(proc, session_id):
    """Task cancellation: start a task, cancel it, verify cancelled status."""
    print("\n  Scenario: Task Cancellation")
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

    # Read the CancelTask response
    cancel_msgs = read_lines(proc, count=1, timeout_sec=10)

    cancel_response = None
    for msg in cancel_msgs:
        if msg.get("id") == 5 and "result" in msg:
            cancel_response = msg["result"]

    if cancel_response is None:
        # Also check if we got remaining events
        remaining = read_all(proc, timeout_sec=5)
        for msg in remaining:
            if msg.get("id") == 5 and "result" in msg:
                cancel_response = msg["result"]
                break

    if cancel_response is None:
        raise TestFailure("No response received for CancelTask")

    print(f"    Cancel response: {json.dumps(cancel_response)}")

    if cancel_response.get("status") != "cancelled":
        raise TestFailure(f"Expected status='cancelled', got: {cancel_response.get('status')}")

    # Drain any remaining events
    read_all(proc, timeout_sec=3)


def test_session_resume():
    """Session resume: create session, run task, shutdown, resume in new runtime."""
    print("\n  Scenario: Session Resume")

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
        print(f"    Step 1: First runtime shutdown OK")
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
            raise TestFailure(f"Expected at least 1 message from ResumeSession, got {len(messages)}")

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
    """Token budget check: verify GetSessionState returns tokenUsage."""
    print("\n  Scenario: Token Budget Check")
    send(proc, {
        "jsonrpc": "2.0", "id": 6, "method": "GetSessionState", "params": {},
    })

    msgs = read_lines(proc, count=1, timeout_sec=10)
    if not msgs:
        raise TestFailure("No response to GetSessionState")

    response = msgs[0]
    if "error" in response:
        raise TestFailure(f"GetSessionState error: {response['error']}")

    result = response.get("result", {})
    print(f"    Session state: {json.dumps(result, indent=2)}")

    token_usage = result.get("tokenUsage")
    if token_usage is None:
        raise TestFailure("tokenUsage not present in GetSessionState response")

    required_fields = ["inputTokens", "outputTokens", "totalTokens", "remaining", "maxSessionTokens"]
    for field in required_fields:
        if field not in token_usage:
            raise TestFailure(f"tokenUsage missing field: {field}")

    # Verify types are numeric
    for field in required_fields:
        if not isinstance(token_usage[field], (int, float)):
            raise TestFailure(f"tokenUsage.{field} is not numeric: {token_usage[field]}")

    print(f"    Token usage: {token_usage['totalTokens']} used, {token_usage['remaining']} remaining")


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

    scenarios = [
        ("Happy Path", test_happy_path),
        ("LLM Error Simulation", test_llm_error_simulation),
        ("Task Cancellation", test_task_cancellation),
        ("Token Budget Check", test_token_budget_check),
    ]

    # Step 1: Spawn
    print("\n1. Spawning agent-runtime...")
    try:
        proc = spawn_runtime()
    except TestFailure as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    print(f"  OK: Running (pid={proc.pid})")

    # Step 2: CreateSession
    print("\n2. CreateSession...")
    try:
        session_id, session_msgs = create_session(proc)
    except TestFailure as e:
        print(f"FAIL: {e}")
        kill_runtime(proc)
        dump_stderr(proc)
        sys.exit(1)

    for msg in session_msgs:
        if "method" in msg:
            evt = msg.get("params", {}).get("eventType", "")
            print(f"  NOTIFICATION: {msg['method']} -> {evt}")
        else:
            result = msg.get("result", {})
            print(f"  RESPONSE: sessionId={result.get('sessionId', '')[:12]}... status={result.get('status')}")
    print(f"  OK: Session created (id={session_id[:12]}...)")

    # Step 3: Run scenarios
    print("\n3. Running scenarios...")
    passed = 0
    failed = 0
    results = []

    for name, test_fn in scenarios:
        try:
            test_fn(proc, session_id)
            print(f"    PASS: {name}")
            passed += 1
            results.append((name, True, None))
        except TestFailure as e:
            print(f"    FAIL: {name} — {e}")
            failed += 1
            results.append((name, False, str(e)))
        except Exception as e:
            print(f"    FAIL: {name} — unexpected error: {e}")
            failed += 1
            results.append((name, False, str(e)))

    # Step 4: Shutdown
    print("\n4. Shutdown...")
    try:
        shutdown_runtime(proc)
        print(f"  OK: Exit code={proc.returncode}")
    except TestFailure as e:
        print(f"  WARN: Shutdown issue: {e}")
        kill_runtime(proc)

    dump_stderr(proc)

    # Step 5: Session Resume (requires a fresh runtime process)
    print("\n5. Session Resume...")
    try:
        test_session_resume()
        print(f"    PASS: Session Resume")
        passed += 1
        results.append(("Session Resume", True, None))
    except TestFailure as e:
        print(f"    FAIL: Session Resume — {e}")
        failed += 1
        results.append(("Session Resume", False, str(e)))
    except Exception as e:
        print(f"    FAIL: Session Resume — unexpected error: {e}")
        failed += 1
        results.append(("Session Resume", False, str(e)))

    # Summary
    total = len(scenarios) + 1  # +1 for session resume
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
