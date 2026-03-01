"""Windows platform adapter."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


class WindowsAdapter:
    """Platform adapter for Windows."""

    @property
    def path_separator(self) -> str:
        return "\\"

    @property
    def default_encoding(self) -> str:
        return "windows-1252"

    @property
    def shell_executable(self) -> str:
        return "cmd.exe"

    @property
    def shell_flag(self) -> str:
        return "/c"

    @property
    def max_path_length(self) -> int:
        return 260

    def normalize_path(self, path: str) -> str:
        return os.path.normpath(path)

    async def resolve_symlinks(self, path: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, os.path.realpath, path)

    def home_directory(self) -> str:
        return str(Path("~").expanduser())

    async def kill_process_tree(self, pid: int) -> None:
        """Kill process tree using taskkill /T /F on Windows."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "taskkill",
                "/T",
                "/F",
                "/PID",
                str(pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except FileNotFoundError:
            pass

    def normalize_line_endings(self, text: str) -> str:
        return text.replace("\r\n", "\n")
