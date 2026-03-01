#!/usr/bin/env python3
"""
Manual smoke test for the agent-runtime JSON-RPC interface.

Spawns the agent-runtime, sends CreateSession, verifies the response,
then sends Shutdown and confirms clean exit.

Prerequisites:
  - Backend services running (session :8000, policy :8001, workspace :8002)
  - LocalStack running (:4566) with DynamoDB tables created
  - Agent-runtime venv installed (.venv/)

Usage:
  make test-jsonrpc          # via Makefile
  python scripts/test-jsonrpc.py  # direct
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
    "LLM_GATEWAY_ENDPOINT": os.environ.get("LLM_GATEWAY_ENDPOINT", "http://localhost:8080"),
    "LLM_GATEWAY_AUTH_TOKEN": os.environ.get("LLM_GATEWAY_AUTH_TOKEN", "test-token"),
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", "info"),
}

CREATE_SESSION_PARAMS = {
    "tenantId": "dev-tenant",
    "userId": "dev-user",
    "workspaceHint": {"localPaths": ["/tmp/test-project"]},
    "clientInfo": {
        "desktopAppVersion": "0.1.0",
        "localAgentHostVersion": "0.1.0",
    },
    "supportedCapabilities": ["File.Read", "File.Write", "Shell.Exec"],
}


# --- Helpers ---

def send(proc, request):
    """Send a JSON-RPC request to the agent-runtime."""
    data = json.dumps(request) + "\n"
    proc.stdin.write(data.encode())
    proc.stdin.flush()


def read_lines(proc, count, timeout_sec=15):
    """Read up to `count` lines from stdout within timeout."""
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


def print_message(label, msg):
    """Pretty-print a JSON-RPC message."""
    if "method" in msg:
        event_type = msg.get("params", {}).get("eventType", "")
        print(f"  {label} [NOTIFICATION] {msg['method']}: {event_type}")
    else:
        result = msg.get("result", msg.get("error", {}))
        print(f"  {label} [RESPONSE id={msg.get('id')}]: {json.dumps(result)}")


def fail(msg):
    print(f"\nFAIL: {msg}")
    sys.exit(1)


def ok(msg):
    print(f"  OK: {msg}")


# --- Main ---

def main():
    print("=" * 60)
    print("Agent-Runtime JSON-RPC Smoke Test")
    print("=" * 60)

    # Check Python venv exists
    if not os.path.exists(PYTHON):
        fail(f"Python venv not found at {PYTHON}. Run: make install")

    # Spawn agent-runtime
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
        fail(f"Process exited immediately (code={proc.returncode})\n{stderr}")

    ok(f"Running (pid={proc.pid})")

    # Send CreateSession
    print("\n2. Sending CreateSession...")
    send(proc, {"jsonrpc": "2.0", "id": 1, "method": "CreateSession", "params": CREATE_SESSION_PARAMS})

    # Read notification + response
    messages = read_lines(proc, count=2, timeout_sec=15)

    if len(messages) < 2:
        fail(f"Expected 2 messages, got {len(messages)}")

    for i, msg in enumerate(messages):
        print_message(f"Line {i + 1}", msg)

    # Validate notification
    notif = messages[0]
    if notif.get("method") != "SessionEvent":
        fail(f"Expected SessionEvent notification, got: {notif}")
    if notif.get("params", {}).get("eventType") != "session_created":
        fail(f"Expected session_created event, got: {notif.get('params', {}).get('eventType')}")
    ok("SessionEvent notification received")

    # Validate response
    resp = messages[1]
    if resp.get("id") != 1:
        fail(f"Expected response id=1, got id={resp.get('id')}")
    result = resp.get("result", {})
    session_id = result.get("sessionId", "")
    workspace_id = result.get("workspaceId", "")
    status = result.get("status", "")
    if not session_id or not workspace_id:
        fail(f"Missing sessionId or workspaceId in response: {result}")
    if status != "ready":
        fail(f"Expected status=ready, got status={status}")
    ok(f"CreateSession response: session={session_id[:12]}... workspace={workspace_id[:12]}... status={status}")

    # Send Shutdown
    print("\n3. Sending Shutdown...")
    send(proc, {"jsonrpc": "2.0", "id": 99, "method": "Shutdown", "params": {}})

    shutdown_msgs = read_lines(proc, count=1, timeout_sec=10)
    if not shutdown_msgs:
        fail("No shutdown response received")

    shutdown_resp = shutdown_msgs[0]
    print_message("Shutdown", shutdown_resp)

    if shutdown_resp.get("id") != 99:
        fail(f"Expected shutdown response id=99, got id={shutdown_resp.get('id')}")
    ok("Shutdown response received")

    # Wait for clean exit
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        fail("Process did not exit within 10s after Shutdown")

    if proc.returncode != 0:
        fail(f"Process exited with code {proc.returncode} (expected 0)")
    ok(f"Process exited cleanly (code={proc.returncode})")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
