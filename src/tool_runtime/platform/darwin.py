"""macOS (Darwin) platform adapter.

Also used for Linux since the behavior is nearly identical.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path


class DarwinAdapter:
    """Platform adapter for macOS and Linux."""

    @property
    def path_separator(self) -> str:
        return "/"

    @property
    def default_encoding(self) -> str:
        # Fallback encoding for non-UTF-8 files (used in the decode chain:
        # utf-8 → platform default → latin-1). mac_roman for macOS legacy
        # files, latin-1 for Linux.
        if sys.platform == "darwin":
            return "mac_roman"
        return "latin-1"

    @property
    def shell_executable(self) -> str:
        # macOS defaults to zsh; Linux defaults to bash
        if sys.platform == "darwin":
            return "/bin/zsh"
        return "/bin/bash"

    @property
    def shell_flag(self) -> str:
        return "-c"

    @property
    def max_path_length(self) -> int:
        return 1024

    def normalize_path(self, path: str) -> str:
        return os.path.normpath(path)

    async def resolve_symlinks(self, path: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, os.path.realpath, path)

    def home_directory(self) -> str:
        return str(Path("~").expanduser())

    async def kill_process_tree(self, pid: int) -> None:
        """Kill process group: SIGTERM, wait 5s, then SIGKILL if still alive."""
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        await asyncio.sleep(5)

        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)

    def normalize_line_endings(self, text: str) -> str:
        return text
