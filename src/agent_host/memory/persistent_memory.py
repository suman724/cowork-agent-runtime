"""PersistentMemory — AI-writable memory files that persist across sessions."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Maximum lines loaded from MEMORY.md at session start
MAX_INDEX_LINES = 200

# Maximum filename length (excluding extension)
MAX_FILENAME_LENGTH = 64

# Only .md files are allowed in the memory directory
_VALID_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+\.md$")


@dataclass(frozen=True)
class MemoryResult:
    """Structured result from memory file operations."""

    success: bool
    content: str  # file content on read success, confirmation on save/delete, error on failure


class PersistentMemory:
    """Manages AI-writable memory files that persist across sessions.

    Memory directory: ``~/.cowork/projects/<hash>/memory/``
    where ``<hash>`` is derived from the workspace directory path.
    """

    def __init__(
        self,
        memory_dir: str,
        *,
        max_file_size: int = 102_400,
        max_file_count: int = 50,
    ) -> None:
        self._memory_dir = Path(memory_dir)
        self._max_file_size = max_file_size
        self._max_file_count = max_file_count

    @staticmethod
    def resolve_memory_dir(workspace_dir: str) -> str:
        """Compute the memory directory path for a given workspace.

        Uses a SHA-256 hash (first 16 hex chars) of the resolved workspace
        path to create a stable, collision-resistant directory name.
        """
        resolved = str(Path(workspace_dir).resolve())
        path_hash = hashlib.sha256(resolved.encode()).hexdigest()[:16]
        base = _default_memory_base()
        return str(Path(base) / path_hash / "memory")

    def ensure_dir(self) -> None:
        """Create the memory directory if it doesn't exist."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    def load_index(self, max_lines: int = MAX_INDEX_LINES) -> str:
        """Load MEMORY.md, capped at *max_lines*.

        Returns an empty string if the file doesn't exist.
        """
        index_path = self._memory_dir / "MEMORY.md"
        if not index_path.is_file():
            return ""

        try:
            lines = index_path.read_text(encoding="utf-8").splitlines()
            return "\n".join(lines[:max_lines])
        except OSError:
            logger.warning("memory_index_read_failed", path=str(index_path), exc_info=True)
            return ""

    def save_file(self, filename: str, content: str) -> MemoryResult:
        """Write *content* to a memory file.

        Validates the filename (no path traversal, ``.md`` extension only).
        Enforces file size and file count limits.
        Uses atomic write (tempfile + os.replace).
        """
        error = _validate_filename(filename)
        if error:
            return MemoryResult(success=False, content=error)

        # Enforce file size limit
        content_size = len(content.encode("utf-8"))
        if content_size > self._max_file_size:
            return MemoryResult(
                success=False,
                content=(
                    f"Content exceeds max file size ({content_size} > {self._max_file_size} bytes)"
                ),
            )

        self.ensure_dir()
        target = self._memory_dir / filename

        # Enforce file count limit (only for new files, not overwrites)
        if not target.exists():
            existing_count = sum(
                1 for e in self._memory_dir.iterdir() if e.is_file() and e.suffix == ".md"
            )
            if existing_count >= self._max_file_count:
                return MemoryResult(
                    success=False,
                    content=f"Max file count reached ({self._max_file_count})",
                )

        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(self._memory_dir), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                Path(tmp_path).replace(target)
            except BaseException:
                tmp = Path(tmp_path)
                if tmp.exists():
                    tmp.unlink()
                raise
        except OSError:
            logger.warning("memory_save_failed", filename=filename, exc_info=True)
            return MemoryResult(success=False, content=f"Failed to write {filename}")

        logger.info("memory_saved", filename=filename, size=len(content))
        return MemoryResult(success=True, content=f"Saved {filename} ({len(content)} characters)")

    def read_file(self, filename: str) -> MemoryResult:
        """Read a memory file."""
        error = _validate_filename(filename)
        if error:
            return MemoryResult(success=False, content=error)

        filepath = self._memory_dir / filename
        if not filepath.is_file():
            return MemoryResult(success=False, content=f"{filename} not found")

        try:
            content = filepath.read_text(encoding="utf-8")
            return MemoryResult(success=True, content=content)
        except OSError:
            logger.warning("memory_read_failed", filename=filename, exc_info=True)
            return MemoryResult(success=False, content=f"Failed to read {filename}")

    def delete_file(self, filename: str) -> MemoryResult:
        """Delete a memory file. Cannot delete MEMORY.md (use SaveMemory to overwrite instead)."""
        error = _validate_filename(filename)
        if error:
            return MemoryResult(success=False, content=error)
        if filename == "MEMORY.md":
            return MemoryResult(
                success=False,
                content="Cannot delete MEMORY.md — use SaveMemory to overwrite it",
            )
        filepath = self._memory_dir / filename
        if not filepath.is_file():
            return MemoryResult(success=False, content=f"{filename} not found")
        try:
            filepath.unlink()
            logger.info("memory_deleted", filename=filename)
            return MemoryResult(success=True, content=f"Deleted {filename}")
        except OSError:
            logger.warning("memory_delete_failed", filename=filename, exc_info=True)
            return MemoryResult(success=False, content=f"Failed to delete {filename}")

    def list_files(self) -> list[dict[str, str | int]]:
        """List all ``.md`` files in the memory directory with sizes."""
        if not self._memory_dir.is_dir():
            return []

        result: list[dict[str, str | int]] = []
        for entry in sorted(self._memory_dir.iterdir()):
            if entry.is_file() and entry.suffix == ".md":
                result.append({"name": entry.name, "size": entry.stat().st_size})
        return result


def _validate_filename(filename: str) -> str:
    """Return an error message if the filename is invalid, else empty string."""
    if not filename:
        return "Filename is required"
    if "/" in filename or "\\" in filename or ".." in filename:
        return "Path traversal not allowed in filename"
    if len(filename) > MAX_FILENAME_LENGTH:
        return f"Filename too long (max {MAX_FILENAME_LENGTH} characters)"
    if not _VALID_FILENAME_RE.match(filename):
        return "Filename must match [a-zA-Z0-9_-]+.md"
    return ""


def _default_memory_base() -> str:
    """Return the platform-specific base directory for memory storage."""
    system = platform.system()
    if system == "Darwin":
        return str(Path.home() / ".cowork" / "projects")
    if system == "Windows":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return str(Path(appdata) / "cowork" / "projects")
    # Linux and others
    return str(Path.home() / ".cowork" / "projects")
