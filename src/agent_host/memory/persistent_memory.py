"""PersistentMemory — AI-writable memory files that persist across sessions."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Maximum lines loaded from MEMORY.md at session start
MAX_INDEX_LINES = 200

# Only .md files are allowed in the memory directory
_VALID_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+\.md$")


class PersistentMemory:
    """Manages AI-writable memory files that persist across sessions.

    Memory directory: ``~/.cowork/projects/<hash>/memory/``
    where ``<hash>`` is derived from the workspace directory path.
    """

    def __init__(self, memory_dir: str) -> None:
        self._memory_dir = Path(memory_dir)

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

    def save_file(self, filename: str, content: str) -> str:
        """Write *content* to a memory file.  Returns a confirmation message.

        Validates the filename (no path traversal, ``.md`` extension only).
        Uses atomic write (tempfile + os.replace).
        """
        error = _validate_filename(filename)
        if error:
            return error

        self.ensure_dir()
        target = self._memory_dir / filename

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
            return f"Error: failed to write {filename}"

        logger.info("memory_saved", filename=filename, size=len(content))
        return f"Saved {filename} ({len(content)} characters)"

    def read_file(self, filename: str) -> str:
        """Read a memory file.  Returns content or an error message."""
        error = _validate_filename(filename)
        if error:
            return error

        filepath = self._memory_dir / filename
        if not filepath.is_file():
            return f"Error: {filename} not found"

        try:
            return filepath.read_text(encoding="utf-8")
        except OSError:
            logger.warning("memory_read_failed", filename=filename, exc_info=True)
            return f"Error: failed to read {filename}"

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
        return "Error: filename is required"
    if "/" in filename or "\\" in filename or ".." in filename:
        return "Error: path traversal not allowed in filename"
    if not _VALID_FILENAME_RE.match(filename):
        return "Error: filename must match [a-zA-Z0-9_-]+.md"
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
