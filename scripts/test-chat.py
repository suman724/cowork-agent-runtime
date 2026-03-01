#!/usr/bin/env python3
"""
Test a full chat flow: CreateSession → StartTask → stream events → Shutdown.

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


# --- Helpers ---

def send(proc, request):
    data = json.dumps(request) + "\n"
    proc.stdin.write(data.encode())
    proc.stdin.flush()


def read_lines(proc, count, timeout_sec=15):
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


def fail(proc, msg):
    print(f"\nFAIL: {msg}")
    try:
        proc.kill()
        stderr = proc.stderr.read().decode()
        if stderr:
            print(f"\nSTDERR (last 2000 chars):\n{stderr[-2000:]}")
    except Exception:
        pass
    sys.exit(1)


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

    # Step 1: Spawn
    print("\n1. Spawning agent-runtime...")
    proc = subprocess.Popen(
        [PYTHON, "-m", "agent_host.main"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=ENV,
    )
    time.sleep(1)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode()[-1000:]
        print(f"FAIL: Process exited immediately (code={proc.returncode})\n{stderr}")
        sys.exit(1)
    print(f"  OK: Running (pid={proc.pid})")

    # Step 2: CreateSession
    print("\n2. CreateSession...")
    send(proc, {
        "jsonrpc": "2.0", "id": 1, "method": "CreateSession",
        "params": {
            "tenantId": "dev-tenant", "userId": "dev-user",
            "workspaceHint": {"localPaths": ["/tmp/test-chat"]},
            "clientInfo": {"desktopAppVersion": "0.1.0", "localAgentHostVersion": "0.1.0"},
            "supportedCapabilities": ["File.Read", "File.Write", "Shell.Exec"],
        },
    })

    messages = read_lines(proc, count=2, timeout_sec=15)
    if len(messages) < 2:
        fail(proc, f"Expected 2 messages from CreateSession, got {len(messages)}")

    session_id = None
    for msg in messages:
        if "method" in msg:
            evt = msg.get("params", {}).get("eventType", "")
            print(f"  NOTIFICATION: {msg['method']} -> {evt}")
        else:
            result = msg.get("result", {})
            session_id = result.get("sessionId")
            print(f"  RESPONSE: sessionId={result.get('sessionId', '')[:12]}... status={result.get('status')}")

    if not session_id:
        fail(proc, "No sessionId received")

    # Step 3: StartTask
    print("\n3. StartTask (prompt: 'Say hello in one sentence')...")
    send(proc, {
        "jsonrpc": "2.0", "id": 2, "method": "StartTask",
        "params": {
            "sessionId": session_id,
            "taskId": "task-test-1",
            "prompt": "Say hello in one sentence. Do not use any tools.",
            "taskOptions": {"maxSteps": 5},
        },
    })

    print("  Waiting for events...")
    all_msgs = read_all(proc, timeout_sec=60)

    text_chunks = []
    events_seen = []
    has_error = False

    for msg in all_msgs:
        if "method" in msg:
            params = msg.get("params", {})
            evt = params.get("eventType", "")
            events_seen.append(evt)
            if evt == "text_chunk":
                text = params.get("payload", {}).get("text", "")
                text_chunks.append(text)
                sys.stdout.write(text)
            else:
                print(f"  EVENT: {evt}")
        else:
            resp_id = msg.get("id")
            if msg.get("error"):
                has_error = True
                err = msg["error"]
                print(f"  ERROR (id={resp_id}): {err.get('code')} — {err.get('message')}")
            else:
                print(f"  RESPONSE (id={resp_id}): {json.dumps(msg.get('result', {}))}")

    full_text = "".join(text_chunks)
    print(f"\n\n  Full assistant text: '{full_text}'")
    print(f"  Events seen: {events_seen}")
    print(f"  Total messages: {len(all_msgs)}")

    if has_error:
        fail(proc, "Received error response during StartTask")

    if not full_text:
        fail(proc, "No text_chunk events received — LLM may have failed")

    print("  OK: Chat response received")

    # Step 4: Shutdown
    print("\n4. Shutdown...")
    send(proc, {"jsonrpc": "2.0", "id": 99, "method": "Shutdown", "params": {}})

    shutdown_msgs = read_lines(proc, count=1, timeout_sec=10)
    if shutdown_msgs:
        print(f"  RESPONSE: {json.dumps(shutdown_msgs[0].get('result', {}))}")

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        fail(proc, "Process did not exit within 10s after Shutdown")

    print(f"  OK: Exit code={proc.returncode}")

    # Check stderr for errors
    stderr = proc.stderr.read().decode()
    if any(kw in stderr.lower() for kw in ["traceback", "exception"]):
        print("\n--- STDERR (errors found) ---")
        for line in stderr.split("\n"):
            if any(kw in line.lower() for kw in ["error", "exception", "traceback", "failed"]):
                print(f"  {line.strip()}")

    print("\n" + "=" * 60)
    print("CHAT FLOW TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
