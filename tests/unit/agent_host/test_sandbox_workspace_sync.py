"""Tests for workspace file sync (individual file CRUD with parallel operations)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from agent_sdk.exceptions import WorkspaceSyncError

from agent_host.sandbox.workspace_sync import (
    download_files,
    download_workspace,
    upload_files,
    upload_workspace,
)

_PATCH_TARGET = "agent_host.sandbox.workspace_sync.httpx.AsyncClient"


def _mock_client_cls(mock_client: AsyncMock) -> MagicMock:
    """Wrap a mock client so it works as an async context manager."""
    mock_cls = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_cls.return_value = mock_client
    return mock_cls


# -- download_workspace tests --


@pytest.mark.asyncio
async def test_download_workspace_success(tmp_path: Path) -> None:
    """List files then download each one in parallel."""
    list_response = httpx.Response(
        200,
        json=[
            {"path": "hello.txt", "size": 13},
            {"path": "src/main.py", "size": 11},
        ],
    )
    file_responses = {
        "hello.txt": httpx.Response(200, content=b"Hello, world!"),
        "src/main.py": httpx.Response(200, content=b"print('hi')"),
    }

    mock_client = AsyncMock()

    async def mock_get(url: str, **kwargs: object) -> httpx.Response:
        if url.endswith("/files"):
            return list_response
        for file_path, resp in file_responses.items():
            if url.endswith(f"/files/{file_path}"):
                return resp
        return httpx.Response(404)

    mock_client.get = AsyncMock(side_effect=mock_get)

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await download_workspace("http://ws:8000", "ws-123", str(tmp_path))

    assert (tmp_path / "hello.txt").read_text() == "Hello, world!"
    assert (tmp_path / "src" / "main.py").read_text() == "print('hi')"


@pytest.mark.asyncio
async def test_download_workspace_404(tmp_path: Path) -> None:
    """404 on list means empty workspace — no-op."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=httpx.Response(404, text="Not Found"),
    )

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await download_workspace("http://ws:8000", "ws-123", str(tmp_path))

    assert tmp_path.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_download_workspace_empty_list(tmp_path: Path) -> None:
    """Empty file list is a no-op."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=httpx.Response(200, json=[]),
    )

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await download_workspace("http://ws:8000", "ws-123", str(tmp_path))

    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_download_workspace_server_error(tmp_path: Path) -> None:
    """Server error on list raises WorkspaceSyncError."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=httpx.Response(500, text="Internal Server Error"),
    )

    with (
        patch(_PATCH_TARGET, _mock_client_cls(mock_client)),
        pytest.raises(WorkspaceSyncError, match="HTTP 500"),
    ):
        await download_workspace("http://ws:8000", "ws-123", str(tmp_path))


@pytest.mark.asyncio
async def test_download_workspace_http_error(tmp_path: Path) -> None:
    """Connection error on list raises WorkspaceSyncError."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        side_effect=httpx.ConnectError("refused"),
    )

    with (
        patch(_PATCH_TARGET, _mock_client_cls(mock_client)),
        pytest.raises(WorkspaceSyncError, match="Failed to list"),
    ):
        await download_workspace("http://ws:8000", "ws-123", str(tmp_path))


@pytest.mark.asyncio
async def test_download_workspace_path_traversal(tmp_path: Path) -> None:
    """Files with path traversal are skipped."""
    list_response = httpx.Response(
        200,
        json=[
            {"path": "../../../etc/passwd", "size": 6},
            {"path": "safe.txt", "size": 4},
        ],
    )

    mock_client = AsyncMock()

    async def mock_get(url: str, **kwargs: object) -> httpx.Response:
        if url.endswith("/files"):
            return list_response
        if url.endswith("/files/safe.txt"):
            return httpx.Response(200, content=b"safe")
        return httpx.Response(200, content=b"danger")

    mock_client.get = AsyncMock(side_effect=mock_get)

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await download_workspace("http://ws:8000", "ws-123", str(tmp_path))

    assert (tmp_path / "safe.txt").read_text() == "safe"


@pytest.mark.asyncio
async def test_download_workspace_partial_failure(
    tmp_path: Path,
) -> None:
    """Some file downloads fail — others still succeed."""
    list_response = httpx.Response(
        200,
        json=[
            {"path": "good.txt", "size": 4},
            {"path": "bad.txt", "size": 3},
        ],
    )

    mock_client = AsyncMock()

    async def mock_get(url: str, **kwargs: object) -> httpx.Response:
        if url.endswith("/files"):
            return list_response
        if url.endswith("/files/good.txt"):
            return httpx.Response(200, content=b"good")
        if url.endswith("/files/bad.txt"):
            return httpx.Response(500, text="error")
        return httpx.Response(404)

    mock_client.get = AsyncMock(side_effect=mock_get)

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await download_workspace("http://ws:8000", "ws-123", str(tmp_path))

    assert (tmp_path / "good.txt").read_text() == "good"
    assert not (tmp_path / "bad.txt").exists()


@pytest.mark.asyncio
async def test_download_workspace_all_files_fail(
    tmp_path: Path,
) -> None:
    """If every file download fails, raise WorkspaceSyncError."""
    list_response = httpx.Response(
        200,
        json=[{"path": "a.txt", "size": 1}],
    )

    mock_client = AsyncMock()

    async def mock_get(url: str, **kwargs: object) -> httpx.Response:
        if url.endswith("/files"):
            return list_response
        return httpx.Response(500, text="error")

    mock_client.get = AsyncMock(side_effect=mock_get)

    with (
        patch(_PATCH_TARGET, _mock_client_cls(mock_client)),
        pytest.raises(WorkspaceSyncError, match="All file downloads"),
    ):
        await download_workspace("http://ws:8000", "ws-123", str(tmp_path))


# -- upload_workspace tests --


@pytest.mark.asyncio
async def test_upload_workspace_success(tmp_path: Path) -> None:
    """Upload individual files via POST with path query param."""
    (tmp_path / "hello.txt").write_text("hello")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=httpx.Response(200, json={"status": "ok"}),
    )

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await upload_workspace("http://ws:8000", "ws-123", str(tmp_path))

    assert mock_client.post.call_count == 2

    paths_uploaded = set()
    for call in mock_client.post.call_args_list:
        params = call[1].get("params", {})
        paths_uploaded.add(params.get("path"))
    assert paths_uploaded == {"hello.txt", "src/main.py"}


@pytest.mark.asyncio
async def test_upload_workspace_missing_dir() -> None:
    """Missing source dir is a no-op."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock()

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await upload_workspace(
            "http://ws:8000",
            "ws-123",
            "/nonexistent/dir",
        )

    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_upload_workspace_empty_dir(tmp_path: Path) -> None:
    """Empty dir is a no-op."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock()

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await upload_workspace("http://ws:8000", "ws-123", str(tmp_path))

    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_upload_workspace_skips_excluded_dirs(
    tmp_path: Path,
) -> None:
    """Should skip .git, __pycache__, node_modules, .venv, etc."""
    (tmp_path / "keep.txt").write_text("keep")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git config")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "module.pyc").write_bytes(b"\x00")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("module")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("python")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=httpx.Response(200, json={"status": "ok"}),
    )

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await upload_workspace("http://ws:8000", "ws-123", str(tmp_path))

    assert mock_client.post.call_count == 1
    call_kwargs = mock_client.post.call_args[1]
    assert call_kwargs["params"]["path"] == "keep.txt"


@pytest.mark.asyncio
async def test_upload_workspace_server_error_best_effort(
    tmp_path: Path,
) -> None:
    """Server error on upload should not raise — best-effort."""
    (tmp_path / "file.txt").write_text("content")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=httpx.Response(500, text="Internal Server Error"),
    )

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await upload_workspace("http://ws:8000", "ws-123", str(tmp_path))


@pytest.mark.asyncio
async def test_upload_workspace_connection_error_best_effort(
    tmp_path: Path,
) -> None:
    """Connection error on upload should not raise — best-effort."""
    (tmp_path / "file.txt").write_text("content")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        side_effect=httpx.ConnectError("refused"),
    )

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        await upload_workspace("http://ws:8000", "ws-123", str(tmp_path))


# -- download_files tests --


@pytest.mark.asyncio
async def test_download_files_success(tmp_path: Path) -> None:
    """Download specific files by path."""
    mock_client = AsyncMock()

    async def mock_get(url: str, **kwargs: object) -> httpx.Response:
        if url.endswith("/files/hello.txt"):
            return httpx.Response(200, content=b"Hello!")
        if url.endswith("/files/src/main.py"):
            return httpx.Response(200, content=b"print('hi')")
        return httpx.Response(404)

    mock_client.get = AsyncMock(side_effect=mock_get)

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        result = await download_files(
            "http://ws:8000", "ws-123", str(tmp_path), ["hello.txt", "src/main.py"]
        )

    assert set(result["synced"]) == {"hello.txt", "src/main.py"}
    assert result["failed"] == []
    assert (tmp_path / "hello.txt").read_text() == "Hello!"
    assert (tmp_path / "src" / "main.py").read_text() == "print('hi')"


@pytest.mark.asyncio
async def test_download_files_empty_paths(tmp_path: Path) -> None:
    """Empty paths list is a no-op."""
    result = await download_files("http://ws:8000", "ws-123", str(tmp_path), [])
    assert result == {"synced": [], "failed": []}


@pytest.mark.asyncio
async def test_download_files_missing_file(tmp_path: Path) -> None:
    """Missing files in S3 are reported as failed, not raised."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=httpx.Response(404, text="Not Found"),
    )

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        result = await download_files("http://ws:8000", "ws-123", str(tmp_path), ["missing.txt"])

    assert result["synced"] == []
    assert result["failed"] == ["missing.txt"]


@pytest.mark.asyncio
async def test_download_files_path_traversal(tmp_path: Path) -> None:
    """Path traversal attempts are rejected and reported as failed."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=httpx.Response(200, content=b"evil"),
    )

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        result = await download_files(
            "http://ws:8000", "ws-123", str(tmp_path), ["../../../etc/passwd"]
        )

    assert result["synced"] == []
    assert result["failed"] == ["../../../etc/passwd"]


@pytest.mark.asyncio
async def test_download_files_connection_error(tmp_path: Path) -> None:
    """Connection errors are caught and reported as failed."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        result = await download_files("http://ws:8000", "ws-123", str(tmp_path), ["fail.txt"])

    assert result["synced"] == []
    assert result["failed"] == ["fail.txt"]


# -- upload_files tests --


@pytest.mark.asyncio
async def test_upload_files_success(tmp_path: Path) -> None:
    """Upload specific files by path."""
    (tmp_path / "a.txt").write_text("aaa")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("bbb")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=httpx.Response(200, json={"status": "ok"}),
    )

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        result = await upload_files(
            "http://ws:8000", "ws-123", str(tmp_path), ["a.txt", "sub/b.txt"]
        )

    assert set(result["synced"]) == {"a.txt", "sub/b.txt"}
    assert result["failed"] == []
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_upload_files_empty_paths(tmp_path: Path) -> None:
    """Empty paths list is a no-op."""
    result = await upload_files("http://ws:8000", "ws-123", str(tmp_path), [])
    assert result == {"synced": [], "failed": []}


@pytest.mark.asyncio
async def test_upload_files_missing_local(tmp_path: Path) -> None:
    """Files that don't exist locally are reported as failed."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock()

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        result = await upload_files("http://ws:8000", "ws-123", str(tmp_path), ["nonexistent.txt"])

    assert result["synced"] == []
    assert result["failed"] == ["nonexistent.txt"]
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_upload_files_server_error(tmp_path: Path) -> None:
    """Server errors are caught and reported as failed."""
    (tmp_path / "fail.txt").write_text("content")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=httpx.Response(500, text="Error"),
    )

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        result = await upload_files("http://ws:8000", "ws-123", str(tmp_path), ["fail.txt"])

    assert result["synced"] == []
    assert result["failed"] == ["fail.txt"]


@pytest.mark.asyncio
async def test_upload_files_connection_error(tmp_path: Path) -> None:
    """Connection errors are caught and reported as failed."""
    (tmp_path / "fail.txt").write_text("content")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

    with patch(_PATCH_TARGET, _mock_client_cls(mock_client)):
        result = await upload_files("http://ws:8000", "ws-123", str(tmp_path), ["fail.txt"])

    assert result["synced"] == []
    assert result["failed"] == ["fail.txt"]
