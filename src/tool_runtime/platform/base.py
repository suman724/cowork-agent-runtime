"""Platform adapter protocol for OS-specific behavior."""

from __future__ import annotations

from typing import Protocol


class PlatformAdapter(Protocol):
    """Abstracts macOS vs Windows differences for the tool runtime."""

    @property
    def path_separator(self) -> str: ...

    @property
    def default_encoding(self) -> str: ...

    @property
    def shell_executable(self) -> str: ...

    @property
    def shell_flag(self) -> str: ...

    @property
    def max_path_length(self) -> int: ...

    def normalize_path(self, path: str) -> str:
        """Normalize path separators and resolve . / .. components."""
        ...

    async def resolve_symlinks(self, path: str) -> str:
        """Resolve symlinks to get the real path (may do I/O)."""
        ...

    def home_directory(self) -> str:
        """Return the user's home directory."""
        ...

    async def kill_process_tree(self, pid: int) -> None:
        """Kill a process and all its children."""
        ...

    def normalize_line_endings(self, text: str) -> str:
        """Normalize line endings to \\n."""
        ...
