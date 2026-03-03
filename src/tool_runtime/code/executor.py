"""PythonExecutor — runs Python scripts as subprocesses."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from tool_runtime.code.models import CodeExecutionResult
from tool_runtime.code.preamble import PREAMBLE

if TYPE_CHECKING:
    from tool_runtime.platform.base import PlatformAdapter

logger = structlog.get_logger(__name__)

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".svg"})


class PythonExecutor:
    """Executes Python scripts as subprocesses."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    async def execute(
        self,
        code: str,
        *,
        working_directory: str | None = None,
        timeout: int = 120,
    ) -> CodeExecutionResult:
        """Write code to temp file, run as subprocess, return structured result."""
        run_id = uuid.uuid4().hex[:12]
        output_dir = os.path.join(tempfile.gettempdir(), f"cowork-code-{run_id}")
        os.makedirs(output_dir, exist_ok=True)

        script_path = os.path.join(output_dir, "script.py")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(PREAMBLE)
                f.write("\n")
                f.write(code)

            result = await self._run_script(
                script_path,
                output_dir=output_dir,
                working_directory=working_directory,
                timeout=timeout,
            )
            return result
        finally:
            self._cleanup(output_dir)

    async def _run_script(
        self,
        script_path: str,
        *,
        output_dir: str,
        working_directory: str | None,
        timeout: int,
    ) -> CodeExecutionResult:
        """Spawn python3, capture output, enforce timeout."""
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["COWORK_OUTPUT_DIR"] = output_dir

        # Use process group for reliable tree kill on Unix
        preexec = os.setsid if sys.platform != "win32" else None

        start = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory,
            env=env,
            preexec_fn=preexec,
        )

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            timed_out = True
            logger.warning("code_execution_timeout", timeout=timeout, pid=process.pid)
            await self._platform.kill_process_tree(process.pid)
            stdout_bytes = b""
            stderr_bytes = f"Code execution timed out after {timeout}s".encode()

        elapsed = time.monotonic() - start

        stdout = self._platform.normalize_line_endings(
            stdout_bytes.decode("utf-8", errors="replace")
        )
        stderr = self._platform.normalize_line_endings(
            stderr_bytes.decode("utf-8", errors="replace")
        )

        images = self._collect_images(output_dir)

        exit_code = process.returncode if process.returncode is not None else -1

        return CodeExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            images=images,
            execution_time=elapsed,
            timed_out=timed_out,
        )

    @staticmethod
    def _collect_images(output_dir: str) -> list[bytes]:
        """Scan output dir for image files and read them."""
        images: list[bytes] = []
        try:
            for entry in sorted(Path(output_dir).iterdir()):
                if entry.suffix.lower() in _IMAGE_EXTENSIONS and entry.is_file():
                    images.append(entry.read_bytes())
        except OSError:
            pass
        return images

    @staticmethod
    def _cleanup(output_dir: str) -> None:
        """Best-effort cleanup of the temp directory."""
        try:
            import shutil

            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:  # noqa: S110
            pass
