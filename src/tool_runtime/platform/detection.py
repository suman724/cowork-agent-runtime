"""Platform detection factory."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from tool_runtime.platform.darwin import DarwinAdapter
from tool_runtime.platform.windows import WindowsAdapter

if TYPE_CHECKING:
    from tool_runtime.platform.base import PlatformAdapter


def get_platform() -> PlatformAdapter:
    """Auto-detect the current platform and return the appropriate adapter.

    Linux uses the Darwin adapter since the behavior is nearly identical.
    """
    if sys.platform == "win32":
        return WindowsAdapter()
    return DarwinAdapter()
