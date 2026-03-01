"""Artifact extraction for large tool outputs.

Outputs exceeding ARTIFACT_THRESHOLD are extracted into ArtifactData
so they can be uploaded to the Workspace Service by agent_host.
"""

from __future__ import annotations

from tool_runtime.models import ARTIFACT_THRESHOLD_BYTES, ArtifactData, RawToolOutput
from tool_runtime.output.formatter import truncate_output


def maybe_extract_artifact(
    output_text: str,
    artifact_type: str,
    artifact_name: str,
    max_output_bytes: int | None = None,
) -> RawToolOutput:
    """If output exceeds the artifact threshold, extract full output as artifact data.

    Returns a RawToolOutput with:
    - output_text: truncated version (for the LLM to see)
    - artifact_data: full output as bytes (for upload to Workspace Service), or None
    """
    encoded = output_text.encode("utf-8", errors="replace")

    if len(encoded) <= ARTIFACT_THRESHOLD_BYTES:
        return RawToolOutput(output_text=output_text)

    artifact_data = ArtifactData(
        artifact_type=artifact_type,
        artifact_name=artifact_name,
        data=encoded,
    )

    effective_max = max_output_bytes if max_output_bytes is not None else ARTIFACT_THRESHOLD_BYTES
    truncated = truncate_output(output_text, effective_max)

    return RawToolOutput(output_text=truncated, artifact_data=artifact_data)
