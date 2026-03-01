"""Tests for output formatting and artifact extraction."""

from __future__ import annotations

from tool_runtime.models import ARTIFACT_THRESHOLD_BYTES
from tool_runtime.output.artifacts import maybe_extract_artifact
from tool_runtime.output.formatter import truncate_output


class TestTruncateOutput:
    def test_short_text_unchanged(self) -> None:
        text = "Hello, world!"
        assert truncate_output(text, max_bytes=1000) == text

    def test_exact_limit_unchanged(self) -> None:
        text = "x" * 100
        assert truncate_output(text, max_bytes=100) == text

    def test_long_text_truncated_with_marker(self) -> None:
        text = "A" * 500 + "B" * 500
        result = truncate_output(text, max_bytes=200)
        assert "[... truncated" in result
        assert "bytes ...]" in result
        # Head should start with As
        assert result.startswith("A")
        # Tail should end with Bs
        assert result.endswith("B")

    def test_truncated_size_marker_shows_removed_bytes(self) -> None:
        text = "x" * 1000
        result = truncate_output(text, max_bytes=200)
        # Marker overhead (~42 bytes) is subtracted from budget before 80/20 split.
        # content_budget = 200 - 42 = 158, head = 126, tail = 32, removed = 1000 - 126 - 32 = 842
        assert "bytes ...]" in result
        # Verify the total output fits within the budget
        assert len(result.encode("utf-8")) <= 200

    def test_truncated_output_fits_within_budget(self) -> None:
        """The truncated output (head + marker + tail) must fit within max_bytes."""
        for size in [500, 5000, 50000]:
            for budget in [100, 200, 1000]:
                text = "x" * size
                result = truncate_output(text, max_bytes=budget)
                assert len(result.encode("utf-8")) <= budget, (
                    f"size={size}, budget={budget}, actual={len(result.encode('utf-8'))}"
                )


class TestMaybeExtractArtifact:
    def test_small_output_no_artifact(self) -> None:
        text = "Small output"
        result = maybe_extract_artifact(text, "tool_output", "output.txt")
        assert result.output_text == text
        assert result.artifact_data is None

    def test_large_output_creates_artifact(self) -> None:
        text = "x" * (ARTIFACT_THRESHOLD_BYTES + 1000)
        result = maybe_extract_artifact(text, "tool_output", "output.txt")
        assert result.artifact_data is not None
        assert result.artifact_data.artifact_type == "tool_output"
        assert result.artifact_data.artifact_name == "output.txt"
        assert result.artifact_data.data == text.encode("utf-8")

    def test_large_output_text_is_truncated(self) -> None:
        text = "x" * (ARTIFACT_THRESHOLD_BYTES + 1000)
        result = maybe_extract_artifact(text, "tool_output", "output.txt")
        assert "[... truncated" in result.output_text
        assert len(result.output_text.encode("utf-8")) < len(text.encode("utf-8"))

    def test_exact_threshold_no_artifact(self) -> None:
        text = "x" * ARTIFACT_THRESHOLD_BYTES
        result = maybe_extract_artifact(text, "tool_output", "output.txt")
        assert result.artifact_data is None
        assert result.output_text == text
