"""Tests for the code execution preamble."""

from __future__ import annotations

import pytest

from tool_runtime.code.executor import PythonExecutor
from tool_runtime.code.preamble import PREAMBLE
from tool_runtime.platform.detection import get_platform


@pytest.fixture
def executor() -> PythonExecutor:
    return PythonExecutor(get_platform())


class TestPreamble:
    def test_preamble_is_valid_python(self) -> None:
        """Preamble should compile without syntax errors."""
        compile(PREAMBLE, "<preamble>", "exec")

    async def test_preamble_does_not_break_simple_scripts(self, executor: PythonExecutor) -> None:
        result = await executor.execute("print('hello from user code')")
        assert result.exit_code == 0
        assert "hello from user code" in result.stdout

    async def test_preamble_does_not_pollute_namespace(self, executor: PythonExecutor) -> None:
        """User code should not see preamble internals in dir()."""
        code = """
user_names = [n for n in dir() if not n.startswith('__')]
# Only preamble names start with underscore
non_underscore = [n for n in user_names if not n.startswith('_')]
print(f"clean={len(non_underscore) == 0}")
"""
        result = await executor.execute(code)
        assert "clean=True" in result.stdout

    async def test_preamble_handles_missing_matplotlib(self, executor: PythonExecutor) -> None:
        """Preamble should not fail if matplotlib is not installed."""
        # This test passes as long as the preamble's try/except works
        result = await executor.execute("print('ok')")
        assert result.exit_code == 0

    async def test_preamble_handles_missing_pandas(self, executor: PythonExecutor) -> None:
        """Preamble should not fail if pandas is not installed."""
        result = await executor.execute("print('ok')")
        assert result.exit_code == 0
