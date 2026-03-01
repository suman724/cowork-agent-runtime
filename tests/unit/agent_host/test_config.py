"""Tests for AgentHostConfig loading from environment variables."""

from __future__ import annotations

import pytest

from agent_host.config import AgentHostConfig, _default_checkpoint_dir


class TestDefaultCheckpointDir:
    def test_returns_string(self) -> None:
        """Returns a non-empty string path."""
        result = _default_checkpoint_dir()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "checkpoints" in result


class TestAgentHostConfig:
    def test_from_env_with_all_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loads all required and optional env vars."""
        monkeypatch.setenv("LLM_GATEWAY_ENDPOINT", "https://llm.test.com")
        monkeypatch.setenv("LLM_GATEWAY_AUTH_TOKEN", "secret-token")
        monkeypatch.setenv("SESSION_SERVICE_URL", "https://sessions.test.com")
        monkeypatch.setenv("WORKSPACE_SERVICE_URL", "https://workspace.test.com")
        monkeypatch.setenv("CHECKPOINT_DIR", "/tmp/test-checkpoints")
        monkeypatch.setenv("APPROVAL_TIMEOUT_SECONDS", "600")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")

        config = AgentHostConfig.from_env()

        assert config.llm_gateway_endpoint == "https://llm.test.com"
        assert config.llm_gateway_auth_token == "secret-token"  # noqa: S105
        assert config.session_service_url == "https://sessions.test.com"
        assert config.workspace_service_url == "https://workspace.test.com"
        assert config.checkpoint_dir == "/tmp/test-checkpoints"
        assert config.approval_timeout_seconds == 600
        assert config.log_level == "debug"
        assert config.llm_model == "openai/gpt-4o-mini"

    def test_from_env_with_required_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uses defaults for optional vars."""
        monkeypatch.setenv("LLM_GATEWAY_ENDPOINT", "https://llm.test.com")
        monkeypatch.setenv("LLM_GATEWAY_AUTH_TOKEN", "token")
        monkeypatch.setenv("SESSION_SERVICE_URL", "https://sessions.test.com")
        monkeypatch.setenv("WORKSPACE_SERVICE_URL", "https://workspace.test.com")
        # Clear optional vars
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
        monkeypatch.delenv("APPROVAL_TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        config = AgentHostConfig.from_env()

        assert config.approval_timeout_seconds == 300
        assert config.log_level == "info"
        assert config.llm_model == "openai/gpt-4o"
        assert "checkpoints" in config.checkpoint_dir

    def test_from_env_missing_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when required env var is missing."""
        monkeypatch.delenv("LLM_GATEWAY_ENDPOINT", raising=False)
        monkeypatch.delenv("LLM_GATEWAY_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("SESSION_SERVICE_URL", raising=False)
        monkeypatch.delenv("WORKSPACE_SERVICE_URL", raising=False)

        with pytest.raises(ValueError, match="LLM_GATEWAY_ENDPOINT"):
            AgentHostConfig.from_env()

    def test_from_env_empty_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when required env var is empty string."""
        monkeypatch.setenv("LLM_GATEWAY_ENDPOINT", "")
        monkeypatch.setenv("LLM_GATEWAY_AUTH_TOKEN", "token")
        monkeypatch.setenv("SESSION_SERVICE_URL", "https://sessions.test.com")
        monkeypatch.setenv("WORKSPACE_SERVICE_URL", "https://workspace.test.com")

        with pytest.raises(ValueError, match="LLM_GATEWAY_ENDPOINT"):
            AgentHostConfig.from_env()

    def test_frozen(self) -> None:
        """Config is immutable (frozen dataclass)."""
        config = AgentHostConfig(
            llm_gateway_endpoint="https://llm.test.com",
            llm_gateway_auth_token="token",
            session_service_url="https://sessions.test.com",
            workspace_service_url="https://workspace.test.com",
        )
        with pytest.raises(AttributeError):
            config.llm_model = "other"  # type: ignore[misc]
