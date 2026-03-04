"""Agent Host configuration from environment variables."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path


def _default_checkpoint_dir() -> str:
    """Return platform-appropriate checkpoint directory."""
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "cowork"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        base = Path(appdata) / "cowork"
    else:
        # Linux / other
        base = Path.home() / ".local" / "share" / "cowork"
    return str(base / "agent-runtime" / "checkpoints")


def _default_log_dir() -> str:
    """Return platform-appropriate log directory."""
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Logs" / "cowork"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        base = Path(appdata) / "cowork" / "agent-runtime" / "logs"
        return str(base)
    else:
        # Linux / other
        base = Path.home() / ".local" / "state" / "cowork"
    return str(base / "agent-runtime")


@dataclass(frozen=True)
class AgentHostConfig:
    """Configuration for the Local Agent Host, loaded from environment variables."""

    # Required
    llm_gateway_endpoint: str
    llm_gateway_auth_token: str
    session_service_url: str
    workspace_service_url: str

    # Optional with defaults
    checkpoint_dir: str = field(default_factory=_default_checkpoint_dir)
    log_dir: str = field(default_factory=_default_log_dir)
    approval_timeout_seconds: int = 300
    log_level: str = "info"
    llm_model: str = "openai/gpt-4o"
    default_max_steps: int = 50
    max_context_tokens: int = 100_000
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 30.0
    recency_window: int = 20
    workspace_sync_interval: int = 5  # Sync to workspace every N steps (0 = disabled)
    memory_max_file_size: int = 102_400  # 100 KB per memory file
    memory_max_file_count: int = 50  # Max memory files per workspace

    @classmethod
    def from_env(cls) -> AgentHostConfig:
        """Load configuration from environment variables.

        Required env vars:
            LLM_GATEWAY_ENDPOINT
            LLM_GATEWAY_AUTH_TOKEN
            SESSION_SERVICE_URL
            WORKSPACE_SERVICE_URL

        Optional env vars:
            CHECKPOINT_DIR
            LOG_DIR
            APPROVAL_TIMEOUT_SECONDS (default: 300)
            LOG_LEVEL (default: info)
            LLM_MODEL (default: openai/gpt-4o)
        """

        def _require(name: str) -> str:
            value = os.environ.get(name)
            if not value:
                msg = f"Required environment variable {name} is not set"
                raise ValueError(msg)
            return value

        return cls(
            llm_gateway_endpoint=_require("LLM_GATEWAY_ENDPOINT"),
            llm_gateway_auth_token=_require("LLM_GATEWAY_AUTH_TOKEN"),
            session_service_url=_require("SESSION_SERVICE_URL"),
            workspace_service_url=_require("WORKSPACE_SERVICE_URL"),
            checkpoint_dir=os.environ.get("CHECKPOINT_DIR", _default_checkpoint_dir()),
            log_dir=os.environ.get("LOG_DIR", _default_log_dir()),
            approval_timeout_seconds=int(os.environ.get("APPROVAL_TIMEOUT_SECONDS", "300")),
            log_level=os.environ.get("LOG_LEVEL", "info"),
            llm_model=os.environ.get("LLM_MODEL", "openai/gpt-4o"),
            default_max_steps=int(os.environ.get("DEFAULT_MAX_STEPS", "50")),
            max_context_tokens=int(os.environ.get("MAX_CONTEXT_TOKENS", "100000")),
            llm_max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
            llm_retry_base_delay=float(os.environ.get("LLM_RETRY_BASE_DELAY", "1.0")),
            llm_retry_max_delay=float(os.environ.get("LLM_RETRY_MAX_DELAY", "30.0")),
            recency_window=int(os.environ.get("RECENCY_WINDOW", "20")),
            workspace_sync_interval=int(os.environ.get("WORKSPACE_SYNC_INTERVAL", "5")),
            memory_max_file_size=int(os.environ.get("MEMORY_MAX_FILE_SIZE", "102400")),
            memory_max_file_count=int(os.environ.get("MEMORY_MAX_FILE_COUNT", "50")),
        )
