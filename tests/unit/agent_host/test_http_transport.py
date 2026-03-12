"""Tests for HttpTransport — HTTP/SSE server for web/sandbox mode."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from agent_host.transport.http_transport import HttpTransport
from agent_host.transport.method_dispatcher import MethodDispatcher


@pytest.fixture
def workspace_dir() -> Any:
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def transport(workspace_dir: str) -> HttpTransport:
    """Create an HttpTransport with a mock dispatcher."""
    t = HttpTransport(workspace_dir=workspace_dir)
    dispatcher = MethodDispatcher()
    t.set_dispatcher(dispatcher)
    t.set_ready()
    return t


@pytest.fixture
def client(transport: HttpTransport) -> TestClient:
    """Create a Starlette TestClient."""
    return TestClient(transport.app)


class TestHealthEndpoints:
    def test_health(self, client: TestClient) -> None:
        """GET /health returns 200."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_when_ready(self, client: TestClient) -> None:
        """GET /ready returns 200 when transport is ready."""
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_when_not_ready(self, workspace_dir: str) -> None:
        """GET /ready returns 503 when transport is not ready."""
        t = HttpTransport(workspace_dir=workspace_dir)
        c = TestClient(t.app)
        resp = c.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"


class TestRpcEndpoint:
    def test_rpc_no_dispatcher(self, workspace_dir: str) -> None:
        """POST /rpc returns 503 when no dispatcher set."""
        t = HttpTransport(workspace_dir=workspace_dir)
        c = TestClient(t.app)
        resp = c.post("/rpc", content='{"jsonrpc":"2.0","method":"test","id":1}')
        assert resp.status_code == 503

    def test_rpc_valid_request(self, transport: HttpTransport, client: TestClient) -> None:
        """POST /rpc dispatches to the method handler."""
        dispatcher = MethodDispatcher()

        async def echo_handler(params: dict[str, Any]) -> dict[str, Any]:
            return {"echo": params.get("msg", "")}

        dispatcher.register("Echo", echo_handler)
        transport.set_dispatcher(dispatcher)

        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "Echo",
                "params": {"msg": "hello"},
                "id": 1,
            }
        )
        resp = client.post("/rpc", content=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["echo"] == "hello"
        assert data["id"] == 1

    def test_rpc_parse_error(self, client: TestClient) -> None:
        """POST /rpc with invalid JSON returns parse error."""
        resp = client.post("/rpc", content="not json")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == -32700

    def test_rpc_method_not_found(self, client: TestClient) -> None:
        """POST /rpc with unknown method returns method not found."""
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "NonExistent",
                "params": {},
                "id": 1,
            }
        )
        resp = client.post("/rpc", content=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"]["code"] == -32601


class TestFileUpload:
    def test_upload_file(self, client: TestClient, workspace_dir: str) -> None:
        """POST /upload writes file to workspace."""
        resp = client.post(
            "/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["files"]) == 1
        assert data["files"][0]["path"] == "test.txt"
        assert data["files"][0]["size"] == 11
        assert Path(workspace_dir, "test.txt").read_text() == "hello world"

    def test_upload_multiple_files(self, client: TestClient, workspace_dir: str) -> None:
        """POST /upload handles multiple files."""
        resp = client.post(
            "/upload",
            files=[
                ("file1", ("a.txt", b"aaa", "text/plain")),
                ("file2", ("b.txt", b"bbb", "text/plain")),
            ],
        )
        assert resp.status_code == 200
        assert len(resp.json()["files"]) == 2
        assert resp.json()["totalSize"] == 6

    def test_upload_subdirectory(self, client: TestClient, workspace_dir: str) -> None:
        """POST /upload creates parent directories."""
        resp = client.post(
            "/upload",
            files={"file": ("sub/dir/test.txt", b"nested", "text/plain")},
        )
        assert resp.status_code == 200
        assert Path(workspace_dir, "sub", "dir", "test.txt").read_text() == "nested"

    def test_upload_path_traversal_blocked(self, client: TestClient) -> None:
        """POST /upload blocks path traversal attempts."""
        resp = client.post(
            "/upload",
            files={"file": ("../../../etc/passwd", b"evil", "text/plain")},
        )
        assert resp.status_code == 400
        assert "Invalid file path" in resp.json()["error"]

    def test_upload_no_workspace(self) -> None:
        """POST /upload returns 503 when workspace not configured."""
        t = HttpTransport()
        c = TestClient(t.app)
        resp = c.post("/upload", files={"file": ("test.txt", b"data", "text/plain")})
        assert resp.status_code == 503

    def test_upload_wrong_content_type(self, client: TestClient) -> None:
        """POST /upload requires multipart/form-data."""
        resp = client.post("/upload", content=b"raw data")
        assert resp.status_code == 400


class TestFileDownload:
    def test_download_file(self, client: TestClient, workspace_dir: str) -> None:
        """GET /files/{path} serves workspace files."""
        Path(workspace_dir, "hello.txt").write_text("hello")
        resp = client.get("/files/hello.txt")
        assert resp.status_code == 200
        assert resp.text == "hello"

    def test_download_not_found(self, client: TestClient) -> None:
        """GET /files/{path} returns 404 for missing files."""
        resp = client.get("/files/nope.txt")
        assert resp.status_code == 404

    def test_download_path_traversal(self, client: TestClient, workspace_dir: str) -> None:
        """GET /files/{path} blocks path traversal via symlinks."""
        # Starlette normalizes ../.. in URLs, so test with a symlink instead
        link_path = Path(workspace_dir, "evil_link")
        link_path.symlink_to("/etc/hosts")
        resp = client.get("/files/evil_link")
        # The symlink target is outside workspace — should be blocked
        # Note: Path.resolve() follows symlinks, so the resolved path
        # will be /etc/hosts which is outside workspace_dir
        assert resp.status_code == 400

    def test_download_no_workspace(self) -> None:
        """GET /files/{path} returns 503 when workspace not configured."""
        t = HttpTransport()
        c = TestClient(t.app)
        resp = c.get("/files/test.txt")
        assert resp.status_code == 503


class TestFileList:
    def test_list_files(self, client: TestClient, workspace_dir: str) -> None:
        """GET /files lists workspace contents."""
        Path(workspace_dir, "a.txt").write_text("aaa")
        Path(workspace_dir, "b.txt").write_text("bbb")
        resp = client.get("/files")
        assert resp.status_code == 200
        data = resp.json()
        assert data["totalFiles"] == 2
        paths = {f["path"] for f in data["files"]}
        assert paths == {"a.txt", "b.txt"}

    def test_list_files_archive(self, client: TestClient, workspace_dir: str) -> None:
        """GET /files?archive=true returns a zip."""
        Path(workspace_dir, "test.txt").write_text("content")
        resp = client.get("/files?archive=true")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    def test_list_no_workspace(self) -> None:
        """GET /files returns 503 when workspace not configured."""
        t = HttpTransport()
        c = TestClient(t.app)
        resp = c.get("/files")
        assert resp.status_code == 503


class TestSharedEventBuffer:
    def test_send_event_is_noop(self, transport: HttpTransport) -> None:
        """send_event() is a no-op — events are buffered by EventEmitter."""
        transport.send_event({"type": "test"})
        # Buffer should be empty because send_event is a no-op on HttpTransport
        assert transport.event_buffer.size == 0

    def test_shared_buffer_feeds_sse(self, transport: HttpTransport) -> None:
        """Events pushed to the shared buffer are available via SSE."""
        # Simulate EventEmitter pushing to the shared buffer
        transport.event_buffer.push({"type": "event1", "payload": {"key": "val"}})
        transport.event_buffer.push({"type": "event2"})

        events, _ = transport.event_buffer.get_since(0)
        assert len(events) == 2
        assert events[0].data["type"] == "event1"
        assert events[0].data["payload"] == {"key": "val"}


class TestWorkspaceSyncRpc:
    """Tests for the workspace.sync JSON-RPC handler."""

    @pytest.fixture
    def sync_transport(self, workspace_dir: str) -> HttpTransport:
        """Transport with workspace sync context configured."""
        t = HttpTransport(workspace_dir=workspace_dir)
        dispatcher = MethodDispatcher()
        dispatcher.register("workspace.sync", t.handle_workspace_sync)
        t.set_dispatcher(dispatcher)
        t.set_ready()
        t.set_workspace_sync_context("http://ws:8000", "ws-123")
        t.mark_startup_sync_complete()
        return t

    @pytest.fixture
    def sync_client(self, sync_transport: HttpTransport) -> TestClient:
        return TestClient(sync_transport.app)

    def _rpc(self, client: TestClient, params: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps({"jsonrpc": "2.0", "method": "workspace.sync", "params": params, "id": 1})
        resp = client.post("/rpc", content=body)
        assert resp.status_code == 200
        return resp.json()

    def test_pull_with_paths(self, sync_client: TestClient, workspace_dir: str) -> None:
        """workspace.sync pull with specific paths downloads files."""
        mock_result = {"synced": ["test.txt"], "failed": []}
        with patch(
            "agent_host.sandbox.workspace_sync.download_files",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            data = self._rpc(sync_client, {"direction": "pull", "paths": ["test.txt"]})

        assert "result" in data
        assert data["result"]["direction"] == "pull"
        assert data["result"]["synced"] == ["test.txt"]

    def test_push_with_paths(self, sync_client: TestClient, workspace_dir: str) -> None:
        """workspace.sync push with specific paths uploads files."""
        # Create a file to upload
        Path(workspace_dir, "out.txt").write_text("output")

        mock_result = {"synced": ["out.txt"], "failed": []}
        with patch(
            "agent_host.sandbox.workspace_sync.upload_files",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            data = self._rpc(sync_client, {"direction": "push", "paths": ["out.txt"]})

        assert "result" in data
        assert data["result"]["direction"] == "push"
        assert data["result"]["synced"] == ["out.txt"]

    def test_pull_without_paths(self, sync_client: TestClient) -> None:
        """workspace.sync pull without paths does full download."""
        with patch(
            "agent_host.sandbox.workspace_sync.download_workspace",
            new_callable=AsyncMock,
        ) as mock_dl:
            data = self._rpc(sync_client, {"direction": "pull"})

        mock_dl.assert_called_once()
        assert "result" in data
        assert data["result"]["direction"] == "pull"

    def test_push_without_paths(self, sync_client: TestClient) -> None:
        """workspace.sync push without paths does full upload."""
        with patch(
            "agent_host.sandbox.workspace_sync.upload_workspace",
            new_callable=AsyncMock,
        ) as mock_ul:
            data = self._rpc(sync_client, {"direction": "push"})

        mock_ul.assert_called_once()
        assert "result" in data
        assert data["result"]["direction"] == "push"

    def test_invalid_direction(self, sync_client: TestClient) -> None:
        """workspace.sync with invalid direction returns error."""
        data = self._rpc(sync_client, {"direction": "invalid"})
        assert "error" in data
        assert data["error"]["code"] == -32061

    def test_no_sync_context(self, workspace_dir: str) -> None:
        """workspace.sync without sync context configured returns error."""
        t = HttpTransport(workspace_dir=workspace_dir)
        dispatcher = MethodDispatcher()
        dispatcher.register("workspace.sync", t.handle_workspace_sync)
        t.set_dispatcher(dispatcher)
        t.set_ready()
        t.mark_startup_sync_complete()
        # NOT calling set_workspace_sync_context()

        c = TestClient(t.app)
        body = json.dumps(
            {"jsonrpc": "2.0", "method": "workspace.sync", "params": {"direction": "pull"}, "id": 1}
        )
        resp = c.post("/rpc", content=body)
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == -32061

    def test_startup_gate_blocks(self, workspace_dir: str) -> None:
        """workspace.sync blocks until startup sync completes, times out if not set."""
        t = HttpTransport(workspace_dir=workspace_dir)
        t.STARTUP_SYNC_GATE_TIMEOUT = 0.1  # Very short timeout for test
        dispatcher = MethodDispatcher()
        dispatcher.register("workspace.sync", t.handle_workspace_sync)
        t.set_dispatcher(dispatcher)
        t.set_ready()
        t.set_workspace_sync_context("http://ws:8000", "ws-123")
        # NOT calling mark_startup_sync_complete()

        c = TestClient(t.app)
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "workspace.sync",
                "params": {"direction": "pull", "paths": ["test.txt"]},
                "id": 1,
            }
        )
        resp = c.post("/rpc", content=body)
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == -32061
        assert "not yet complete" in data["error"]["message"]

    def test_no_workspace_dir(self) -> None:
        """workspace.sync without workspace dir returns error."""
        t = HttpTransport()  # No workspace_dir
        dispatcher = MethodDispatcher()
        dispatcher.register("workspace.sync", t.handle_workspace_sync)
        t.set_dispatcher(dispatcher)
        t.set_ready()
        t.set_workspace_sync_context("http://ws:8000", "ws-123")
        t.mark_startup_sync_complete()

        c = TestClient(t.app)
        body = json.dumps(
            {"jsonrpc": "2.0", "method": "workspace.sync", "params": {"direction": "pull"}, "id": 1}
        )
        resp = c.post("/rpc", content=body)
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == -32061
