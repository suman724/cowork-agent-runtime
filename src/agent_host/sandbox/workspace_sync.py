"""Workspace file sync: download on startup, upload on shutdown, targeted sync.

Uses individual file CRUD operations from the Workspace Service:
- GET  /workspaces/{id}/files         → list files
- GET  /workspaces/{id}/files/{path}  → download one file
- POST /workspaces/{id}/files?path=X  → upload one file (multipart)

All file operations run in parallel with bounded concurrency.

Functions:
- download_workspace / upload_workspace — full sync (startup/shutdown)
- download_files / upload_files — targeted sync (individual paths)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import structlog
from agent_sdk.exceptions import WorkspaceSyncError

logger = structlog.get_logger()

# Max concurrent file operations to avoid overwhelming the service
_MAX_CONCURRENCY = 10

# Shared timeout for all workspace sync HTTP calls
_SYNC_TIMEOUT = httpx.Timeout(120.0, connect=10.0)

# Directories to exclude from workspace uploads
_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


def _files_base_url(workspace_service_url: str, workspace_id: str) -> str:
    """Build the base URL for workspace file operations."""
    return f"{workspace_service_url.rstrip('/')}/workspaces/{workspace_id}/files"


async def download_workspace(
    workspace_service_url: str,
    workspace_id: str,
    target_dir: str,
) -> None:
    """Download workspace files from the Workspace Service to a local directory.

    Lists files via GET /workspaces/{id}/files, then downloads each file
    in parallel via GET /workspaces/{id}/files/{path}.

    If the workspace has no files, this is a no-op.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    target_resolved = target.resolve()

    base_url = _files_base_url(workspace_service_url, workspace_id)

    async with httpx.AsyncClient(timeout=_SYNC_TIMEOUT) as client:
        # Step 1: List files
        try:
            resp = await client.get(base_url)
        except httpx.HTTPError as exc:
            raise WorkspaceSyncError(
                f"Failed to list workspace files: {exc}",
            ) from exc

        if resp.status_code == 404:
            logger.info("workspace_empty", workspace_id=workspace_id)
            return

        if resp.status_code >= 400:
            raise WorkspaceSyncError(f"Workspace file list failed (HTTP {resp.status_code})")

        try:
            file_list: list[dict[str, object]] = resp.json()
        except ValueError as exc:
            raise WorkspaceSyncError(
                f"Invalid file list response: {exc}",
            ) from exc

        if not file_list:
            logger.info("workspace_no_files", workspace_id=workspace_id)
            return

        # Step 2: Download each file in parallel
        semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

        async def _download_one(file_path: str) -> bool:
            """Download a single file. Returns True on success."""
            async with semaphore:
                try:
                    dl_resp = await client.get(
                        f"{base_url}/{file_path}",
                    )
                except httpx.HTTPError as exc:
                    logger.warning(
                        "workspace_file_download_failed",
                        workspace_id=workspace_id,
                        path=file_path,
                        error=str(exc),
                    )
                    return False

                if dl_resp.status_code >= 400:
                    logger.warning(
                        "workspace_file_download_error",
                        workspace_id=workspace_id,
                        path=file_path,
                        status=dl_resp.status_code,
                    )
                    return False

                # Validate path doesn't escape target directory
                dest = (target / file_path).resolve()
                if not dest.is_relative_to(target_resolved):
                    logger.warning(
                        "workspace_file_path_traversal",
                        workspace_id=workspace_id,
                        path=file_path,
                    )
                    return False

                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(dl_resp.content)
                return True

        paths = [str(f.get("path", "")) for f in file_list]
        results = await asyncio.gather(
            *[_download_one(p) for p in paths if p],
        )
        succeeded = sum(1 for r in results if r)
        failed = len(results) - succeeded

    logger.info(
        "workspace_downloaded",
        workspace_id=workspace_id,
        target_dir=target_dir,
        file_count=succeeded,
        failed=failed,
    )

    if succeeded == 0 and failed > 0:
        raise WorkspaceSyncError("All file downloads failed")


async def upload_workspace(
    workspace_service_url: str,
    workspace_id: str,
    source_dir: str,
) -> None:
    """Upload workspace files from a local directory to the Workspace Service.

    Walks the source directory (skipping excluded dirs), then uploads each file
    in parallel via POST /workspaces/{id}/files?path=X with multipart form data.

    Best-effort — logs warnings but does not raise on failure.
    """
    source = Path(source_dir)
    if not source.exists():
        logger.info("workspace_upload_skip_missing", source_dir=source_dir)
        return

    # Collect files, pruning excluded directories via os.walk
    files_to_upload: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
        for fname in filenames:
            abs_path = Path(dirpath) / fname
            rel_path = str(abs_path.relative_to(source))
            files_to_upload.append((abs_path, rel_path))

    if not files_to_upload:
        logger.info("workspace_upload_skip_empty", source_dir=source_dir)
        return

    base_url = f"{workspace_service_url.rstrip('/')}/workspaces/{workspace_id}/files"
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
    ) as client:

        async def _upload_one(
            abs_path: Path,
            rel_path: str,
        ) -> bool:
            """Upload a single file. Returns True on success."""
            async with semaphore:
                try:
                    content = abs_path.read_bytes()
                    resp = await client.post(
                        base_url,
                        params={"path": rel_path},
                        files={
                            "file": (
                                abs_path.name,
                                content,
                                "application/octet-stream",
                            ),
                        },
                    )
                    if resp.status_code >= 400:
                        logger.warning(
                            "workspace_file_upload_error",
                            workspace_id=workspace_id,
                            path=rel_path,
                            status=resp.status_code,
                        )
                        return False
                    return True
                except httpx.HTTPError as exc:
                    logger.warning(
                        "workspace_file_upload_failed",
                        workspace_id=workspace_id,
                        path=rel_path,
                        error=str(exc),
                    )
                    return False

        results = await asyncio.gather(
            *[_upload_one(ap, rp) for ap, rp in files_to_upload],
        )
        uploaded = sum(1 for r in results if r)
        failed = len(results) - uploaded

    logger.info(
        "workspace_uploaded",
        workspace_id=workspace_id,
        source_dir=source_dir,
        uploaded=uploaded,
        failed=failed,
    )


async def download_files(
    workspace_service_url: str,
    workspace_id: str,
    target_dir: str,
    paths: list[str],
) -> dict[str, list[str]]:
    """Download specific files from the Workspace Service to a local directory.

    Best-effort: skips files that don't exist or fail to download.
    Returns dict with 'synced' and 'failed' lists.
    """
    if not paths:
        return {"synced": [], "failed": []}

    # Deduplicate paths while preserving order
    unique_paths = list(dict.fromkeys(paths))

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    target_resolved = target.resolve()

    base_url = _files_base_url(workspace_service_url, workspace_id)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

    async with httpx.AsyncClient(timeout=_SYNC_TIMEOUT) as client:

        async def _download_one(file_path: str) -> bool:
            """Download a single file. Returns True on success."""
            async with semaphore:
                try:
                    resp = await client.get(f"{base_url}/{file_path}")
                except httpx.HTTPError as exc:
                    logger.warning(
                        "download_file_failed",
                        workspace_id=workspace_id,
                        path=file_path,
                        error=str(exc),
                    )
                    return False

                if resp.status_code >= 400:
                    logger.warning(
                        "download_file_error",
                        workspace_id=workspace_id,
                        path=file_path,
                        status=resp.status_code,
                    )
                    return False

                # Path traversal prevention
                dest = (target / file_path).resolve()
                if not dest.is_relative_to(target_resolved):
                    logger.warning(
                        "download_file_path_traversal",
                        workspace_id=workspace_id,
                        path=file_path,
                    )
                    return False

                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(resp.content)
                return True

        results = await asyncio.gather(*[_download_one(p) for p in unique_paths])

    synced = [p for p, ok in zip(unique_paths, results, strict=True) if ok]
    failed = [p for p, ok in zip(unique_paths, results, strict=True) if not ok]

    logger.info(
        "files_downloaded",
        workspace_id=workspace_id,
        synced=len(synced),
        failed=len(failed),
    )
    return {"synced": synced, "failed": failed}


async def upload_files(
    workspace_service_url: str,
    workspace_id: str,
    source_dir: str,
    paths: list[str],
) -> dict[str, list[str]]:
    """Upload specific files from a local directory to the Workspace Service.

    Best-effort: skips files that don't exist locally or fail to upload.
    Returns dict with 'synced' and 'failed' lists.
    """
    if not paths:
        return {"synced": [], "failed": []}

    # Deduplicate paths while preserving order
    unique_paths = list(dict.fromkeys(paths))

    source = Path(source_dir)
    base_url = _files_base_url(workspace_service_url, workspace_id)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

    async with httpx.AsyncClient(timeout=_SYNC_TIMEOUT) as client:

        async def _upload_one(rel_path: str) -> bool:
            """Upload a single file. Returns True on success."""
            async with semaphore:
                abs_path = source / rel_path
                try:
                    content = abs_path.read_bytes()
                except OSError as exc:
                    logger.warning(
                        "upload_file_not_found",
                        workspace_id=workspace_id,
                        path=rel_path,
                        error=str(exc),
                    )
                    return False

                try:
                    resp = await client.post(
                        base_url,
                        params={"path": rel_path},
                        files={
                            "file": (
                                abs_path.name,
                                content,
                                "application/octet-stream",
                            ),
                        },
                    )
                    if resp.status_code >= 400:
                        logger.warning(
                            "upload_file_error",
                            workspace_id=workspace_id,
                            path=rel_path,
                            status=resp.status_code,
                        )
                        return False
                    return True
                except httpx.HTTPError as exc:
                    logger.warning(
                        "upload_file_failed",
                        workspace_id=workspace_id,
                        path=rel_path,
                        error=str(exc),
                    )
                    return False

        results = await asyncio.gather(*[_upload_one(p) for p in unique_paths])

    synced = [p for p, ok in zip(unique_paths, results, strict=True) if ok]
    failed = [p for p, ok in zip(unique_paths, results, strict=True) if not ok]

    logger.info(
        "files_uploaded",
        workspace_id=workspace_id,
        synced=len(synced),
        failed=len(failed),
    )
    return {"synced": synced, "failed": failed}
