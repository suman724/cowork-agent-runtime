"""ProjectInstructionsLoader — load COWORK.md files from the workspace directory."""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()

# Instruction file names, checked in order
_INSTRUCTION_FILES = ("COWORK.md", "COWORK.local.md")


class ProjectInstructionsLoader:
    """Load COWORK.md and COWORK.local.md from the workspace directory.

    Only loads from the workspace directory itself — no ancestor walk.
    COWORK.md is loaded first, then COWORK.local.md (if present).
    """

    def load(self, workspace_dir: str) -> str:
        """Load instruction files from *workspace_dir*, return concatenated text.

        Returns an empty string when no files are found.
        """
        ws_path = Path(workspace_dir).resolve()
        if not ws_path.is_dir():
            return ""

        parts: list[str] = []
        for filename in _INSTRUCTION_FILES:
            filepath = ws_path / filename
            if filepath.is_file():
                try:
                    content = filepath.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(content)
                except OSError:
                    logger.warning(
                        "instruction_file_read_failed", path=str(filepath), exc_info=True
                    )

        return "\n\n".join(parts)
