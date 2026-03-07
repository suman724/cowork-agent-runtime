"""SubAgentManager — legacy module, sub-agent spawning moved to LoopRuntime.

Sub-agent spawning is now handled by ``LoopRuntime.spawn_sub_agent()``.
This module is retained for backward compatibility only and will be removed.
"""

from __future__ import annotations

# Maximum concurrent sub-agents (moved to LoopRuntime)
MAX_CONCURRENT = 5

# Sub-agent limits (moved to LoopRuntime)
_SUB_AGENT_MAX_STEPS = 10
_SUB_AGENT_RECENCY_WINDOW = 10
_RESULT_MAX_CHARS = 2000
