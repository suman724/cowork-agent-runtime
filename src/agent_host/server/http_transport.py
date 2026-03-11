"""HTTP/SSE transport for web/sandbox mode.

Provides a Starlette server with:
- POST /rpc — JSON-RPC 2.0 dispatch
- GET /events — SSE event stream with replay
- GET /health — liveness probe
- GET /ready — readiness probe
- POST /upload — file upload to workspace directory
- GET /files/{path} — file download from workspace
- GET /files — list workspace files (or zip archive with ?archive=true)

Implements the ``Transport`` protocol.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
import zipfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from agent_host.server.event_buffer import EventBuffer
from agent_host.server.json_rpc import (
    JsonRpcError,
    JsonRpcResponse,
    parse_request,
    serialize_response,
)

if TYPE_CHECKING:
    from starlette.requests import Request

logger = structlog.get_logger()


class HttpTransport:
    """HTTP/SSE transport for the agent host.

    Satisfies the ``Transport`` protocol. Starts a Starlette/uvicorn ASGI
    server that exposes JSON-RPC via HTTP and events via SSE.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 8080,
        workspace_dir: str | None = None,
        event_buffer: EventBuffer | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._workspace_dir = Path(workspace_dir) if workspace_dir else None
        self._event_buffer = event_buffer or EventBuffer()
        self._dispatcher: Any = None  # Set before start() via set_dispatcher()
        self._ready = False
        self._server: Any = None
        self._serve_task: asyncio.Task[None] | None = None
        self._app = self._create_app()

    def set_dispatcher(self, dispatcher: Any) -> None:
        """Set the MethodDispatcher for JSON-RPC handling.

        Must be called before start().
        """
        self._dispatcher = dispatcher

    def set_ready(self, ready: bool = True) -> None:
        """Mark the transport as ready (SessionManager initialized)."""
        self._ready = ready

    def set_workspace_dir(self, workspace_dir: str) -> None:
        """Set the workspace directory for file operations."""
        self._workspace_dir = Path(workspace_dir)

    @property
    def event_buffer(self) -> EventBuffer:
        """Access the event buffer (for testing)."""
        return self._event_buffer

    @property
    def app(self) -> Starlette:
        """Access the ASGI app (for testing with TestClient)."""
        return self._app

    def _create_app(self) -> Starlette:
        """Create the Starlette ASGI application."""
        routes = [
            Route("/rpc", self._handle_rpc, methods=["POST"]),
            Route("/events", self._handle_events, methods=["GET"]),
            Route("/health", self._handle_health, methods=["GET"]),
            Route("/ready", self._handle_ready, methods=["GET"]),
            Route("/upload", self._handle_upload, methods=["POST"]),
            Route("/files/{path:path}", self._handle_file_download, methods=["GET"]),
            Route("/files", self._handle_file_list, methods=["GET"]),
        ]
        return Starlette(routes=routes)

    async def start(self) -> None:
        """Start the HTTP server in a background task."""
        import uvicorn

        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._server.serve())
        logger.info(
            "http_transport_started",
            host=self._host,
            port=self._port,
        )

    def send_event(self, event: dict[str, Any]) -> None:
        """No-op for HttpTransport.

        Events are already buffered by ``EventEmitter`` into the shared
        ``EventBuffer``. The SSE endpoint reads directly from that buffer.
        This method exists only to satisfy the ``Transport`` protocol.
        """

    async def shutdown(self) -> None:
        """Gracefully shut down the HTTP server."""
        if self._server:
            self._server.should_exit = True
        if self._serve_task:
            self._serve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._serve_task
        logger.info("http_transport_stopped")

    # --- Route handlers ---

    async def _handle_rpc(self, request: Request) -> Response:
        """POST /rpc — JSON-RPC 2.0 dispatch."""
        if self._dispatcher is None:
            return JSONResponse(
                {"error": "Server not initialized"},
                status_code=503,
            )

        try:
            body = await request.body()
            raw = body.decode("utf-8")
        except Exception:
            return JSONResponse(
                {"error": "Failed to read request body"},
                status_code=400,
            )

        try:
            rpc_request = parse_request(raw)
        except JsonRpcError as e:
            error_response = JsonRpcResponse(id=None, error=e)
            return JSONResponse(
                json.loads(serialize_response(error_response)),
                status_code=200,
            )

        response = await self._dispatcher.dispatch(rpc_request)
        return JSONResponse(
            json.loads(serialize_response(response)),
            status_code=200,
        )

    async def _handle_events(self, request: Request) -> StreamingResponse:
        """GET /events — SSE event stream with optional replay.

        Query params:
        - since: replay events after this ID (default: 0 = all buffered events)
        """
        since_str = request.query_params.get("since", "0")
        try:
            since_id = int(since_str)
        except ValueError:
            since_id = 0

        async def event_stream() -> Any:
            try:
                async for event in self._event_buffer.subscribe(since_id):
                    if event.id == -1 and event.data.get("_gap"):
                        gap_data = json.dumps(
                            {
                                "since": event.data["since"],
                                "message": "Some events were evicted from buffer",
                            }
                        )
                        yield f"event: gap\ndata: {gap_data}\n\n"
                        continue

                    data = json.dumps({"id": event.id, **event.data})
                    yield f"id: {event.id}\nevent: session_event\ndata: {data}\n\n"
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("sse_stream_error")
                return

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _handle_health(self, _request: Request) -> JSONResponse:
        """GET /health — liveness probe (always 200)."""
        return JSONResponse({"status": "ok"})

    async def _handle_ready(self, _request: Request) -> JSONResponse:
        """GET /ready — readiness probe (200 when SessionManager ready)."""
        if self._ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "not_ready"}, status_code=503)

    async def _handle_upload(self, request: Request) -> JSONResponse:
        """POST /upload — multipart file upload to workspace directory."""
        if self._workspace_dir is None:
            return JSONResponse(
                {"error": "Workspace directory not configured"},
                status_code=503,
            )

        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            return JSONResponse(
                {"error": "Content-Type must be multipart/form-data"},
                status_code=400,
            )

        try:
            form = await request.form()
        except Exception:
            logger.warning("upload_form_parse_error")
            return JSONResponse(
                {"error": "Failed to parse multipart form data"},
                status_code=400,
            )
        uploaded_files: list[dict[str, str | int]] = []
        total_size = 0

        for key in form:
            upload = form[key]
            if not hasattr(upload, "filename") or not hasattr(upload, "read"):
                continue

            filename: str | None = getattr(upload, "filename", None)
            if not filename:
                continue

            # Path traversal prevention: resolve and verify within workspace root
            target = (self._workspace_dir / filename).resolve()
            if not str(target).startswith(str(self._workspace_dir.resolve())):
                logger.warning(
                    "upload_path_traversal_blocked",
                    filename=filename,
                )
                return JSONResponse(
                    {"error": f"Invalid file path: {filename}"},
                    status_code=400,
                )

            # Create parent directories
            target.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            read_fn = getattr(upload, "read", None)
            if read_fn is None:
                continue
            content: bytes = await read_fn()
            target.write_bytes(content)

            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            file_size = len(content)
            total_size += file_size
            uploaded_files.append(
                {
                    "path": filename,
                    "size": file_size,
                    "contentType": mime_type,
                }
            )

        await form.close()
        return JSONResponse(
            {
                "files": uploaded_files,
                "totalSize": total_size,
            }
        )

    async def _handle_file_download(self, request: Request) -> Response:
        """GET /files/{path} — serve file from workspace directory."""
        if self._workspace_dir is None:
            return JSONResponse(
                {"error": "Workspace directory not configured"},
                status_code=503,
            )

        file_path = request.path_params.get("path", "")
        if not file_path:
            return JSONResponse({"error": "Path required"}, status_code=400)

        # Path traversal prevention (follows symlinks via resolve())
        target = (self._workspace_dir / file_path).resolve()
        if not str(target).startswith(str(self._workspace_dir.resolve())):
            return JSONResponse({"error": "Invalid path"}, status_code=400)

        if not target.is_file():
            return JSONResponse({"error": "File not found"}, status_code=404)

        try:
            mime_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            content = target.read_bytes()
        except OSError:
            logger.warning("file_download_read_error", path=file_path)
            return JSONResponse({"error": "Failed to read file"}, status_code=500)
        return Response(
            content=content,
            media_type=mime_type,
            headers={"Content-Disposition": f'inline; filename="{target.name}"'},
        )

    async def _handle_file_list(self, request: Request) -> Response:
        """GET /files — list or archive workspace files.

        Query params:
        - archive=true: return zip archive of all workspace files
        """
        if self._workspace_dir is None:
            return JSONResponse(
                {"error": "Workspace directory not configured"},
                status_code=503,
            )

        archive = request.query_params.get("archive", "").lower() == "true"

        if archive:
            return self._create_workspace_archive()

        return self._list_workspace_files()

    def _list_workspace_files(self) -> JSONResponse:
        """List all files in the workspace directory."""
        if self._workspace_dir is None:  # pragma: no cover
            return JSONResponse({"error": "No workspace"}, status_code=503)
        files = []
        total_size = 0

        for path in sorted(self._workspace_dir.rglob("*")):
            if path.is_file():
                rel_path = path.relative_to(self._workspace_dir)
                stat = path.stat()
                mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                files.append(
                    {
                        "path": str(rel_path),
                        "size": stat.st_size,
                        "contentType": mime_type,
                        "lastModified": stat.st_mtime,
                    }
                )
                total_size += stat.st_size

        return JSONResponse(
            {
                "files": files,
                "totalFiles": len(files),
                "totalSize": total_size,
            }
        )

    def _create_workspace_archive(self) -> Response:
        """Create a zip archive of all workspace files."""
        if self._workspace_dir is None:  # pragma: no cover
            return JSONResponse({"error": "No workspace"}, status_code=503)
        buffer = BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(self._workspace_dir.rglob("*")):
                if path.is_file():
                    rel_path = path.relative_to(self._workspace_dir)
                    zf.write(path, str(rel_path))

        buffer.seek(0)
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="workspace.zip"'},
        )
