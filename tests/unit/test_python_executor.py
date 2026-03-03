"""Tests for PythonExecutor — subprocess lifecycle, timeout, output capture."""

from __future__ import annotations

import pytest

from tool_runtime.code.executor import PythonExecutor
from tool_runtime.platform.detection import get_platform


@pytest.fixture
def executor() -> PythonExecutor:
    return PythonExecutor(get_platform())


class TestPythonExecutor:
    async def test_simple_print(self, executor: PythonExecutor) -> None:
        result = await executor.execute("print('hello world')")
        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert not result.timed_out

    async def test_stderr_captured(self, executor: PythonExecutor) -> None:
        result = await executor.execute("import sys; sys.stderr.write('oops\\n')")
        assert "oops" in result.stderr

    async def test_nonzero_exit_code(self, executor: PythonExecutor) -> None:
        result = await executor.execute("import sys; sys.exit(42)")
        assert result.exit_code == 42

    async def test_syntax_error(self, executor: PythonExecutor) -> None:
        result = await executor.execute("def bad(")
        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr

    async def test_timeout_kills_process(self, executor: PythonExecutor) -> None:
        result = await executor.execute("import time; time.sleep(60)", timeout=1)
        assert result.timed_out
        assert result.exit_code != 0

    async def test_execution_time_tracked(self, executor: PythonExecutor) -> None:
        result = await executor.execute("print('fast')")
        assert result.execution_time > 0
        assert result.execution_time < 30  # Should be fast

    async def test_working_directory(self, executor: PythonExecutor, tmp_path: object) -> None:
        result = await executor.execute(
            "import os; print(os.getcwd())",
            working_directory=str(tmp_path),
        )
        assert str(tmp_path) in result.stdout

    async def test_multiline_script(self, executor: PythonExecutor) -> None:
        code = """
x = 0
for i in range(10):
    x += i
print(f"sum={x}")
"""
        result = await executor.execute(code)
        assert result.exit_code == 0
        assert "sum=45" in result.stdout

    async def test_no_images_by_default(self, executor: PythonExecutor) -> None:
        result = await executor.execute("print('no images')")
        assert result.images == []

    async def test_cleanup_happens(self, executor: PythonExecutor) -> None:
        """Verify temp dir is cleaned up after execution."""
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.gettempdir())
        before = set(tmp.glob("cowork-code-*"))
        await executor.execute("print('clean')")
        after = set(tmp.glob("cowork-code-*"))
        # No new leftover temp dirs
        assert after == before
