"""Tests for PendingArtifactStore — FIFO artifact bridge."""

from __future__ import annotations

from agent_host.agent.artifact_store import PendingArtifactStore
from tool_runtime.models import ArtifactData


def _make_artifact(name: str = "test") -> ArtifactData:
    return ArtifactData(
        artifact_type="tool_output",
        artifact_name=name,
        data=b"test data",
        media_type="text/plain",
    )


class TestPendingArtifactStore:
    def test_store_and_pop(self) -> None:
        """Store artifacts, pop returns them, second pop returns empty."""
        store = PendingArtifactStore()
        artifacts = [_make_artifact("a1"), _make_artifact("a2")]
        store.store("ReadFile", artifacts)

        result = store.pop("ReadFile")
        assert len(result) == 2
        assert result[0].artifact_name == "a1"

        # Second pop returns empty
        assert store.pop("ReadFile") == []

    def test_pop_empty(self) -> None:
        """Pop with no stored artifacts returns empty list."""
        store = PendingArtifactStore()
        assert store.pop("ReadFile") == []

    def test_multiple_tools(self) -> None:
        """Store for two tools, pop each independently."""
        store = PendingArtifactStore()
        store.store("ReadFile", [_make_artifact("read-artifact")])
        store.store("WriteFile", [_make_artifact("write-artifact")])

        write_artifacts = store.pop("WriteFile")
        assert len(write_artifacts) == 1
        assert write_artifacts[0].artifact_name == "write-artifact"

        read_artifacts = store.pop("ReadFile")
        assert len(read_artifacts) == 1
        assert read_artifacts[0].artifact_name == "read-artifact"
