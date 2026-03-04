"""ProjectInstructionsLoader — load COWORK.md files from the project directory tree."""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()

# Instruction file names, checked in order at each directory level
_INSTRUCTION_FILES = ("COWORK.md", "COWORK.local.md")


class ProjectInstructionsLoader:
    """Load COWORK.md and COWORK.local.md from the project directory tree.

    Walks up from the workspace directory to the filesystem root, collecting
    instruction files.  Ancestors are loaded first (broadest scope), with the
    workspace-level files last (most specific).  Files at the same level are
    ordered: COWORK.md before COWORK.local.md.
    """

    def load(self, workspace_dir: str) -> str:
        """Walk up from *workspace_dir*, collect instruction files, return concatenated text.

        Returns an empty string when no files are found.
        """
        ws_path = Path(workspace_dir).resolve()
        if not ws_path.is_dir():
            return ""

        # Collect (directory, filename) pairs from root → workspace_dir
        collected: list[tuple[Path, str]] = []
        current = ws_path
        ancestors: list[Path] = []
        while True:
            ancestors.append(current)
            parent = current.parent
            if parent == current:
                break  # reached filesystem root
            current = parent

        # Reverse so ancestors come first (broadest → most specific)
        ancestors.reverse()

        for directory in ancestors:
            for filename in _INSTRUCTION_FILES:
                filepath = directory / filename
                if filepath.is_file():
                    collected.append((filepath, filename))

        if not collected:
            return ""

        parts: list[str] = []
        for filepath, _filename in collected:
            try:
                content = filepath.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(f"--- {filepath} ---\n{content}")
            except OSError:
                logger.warning("instruction_file_read_failed", path=str(filepath), exc_info=True)

        return "\n\n".join(parts)
