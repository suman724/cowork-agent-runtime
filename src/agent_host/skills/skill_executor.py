"""SkillExecutor — legacy module, skill execution moved to LoopRuntime.

Skill execution is now handled by ``LoopRuntime.execute_skill()``.
This module is retained for backward compatibility only and will be removed.
"""

from __future__ import annotations

_SKILL_RECENCY_WINDOW = 10
_RESULT_MAX_CHARS = 4000
