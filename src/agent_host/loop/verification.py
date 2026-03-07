"""Verification phase configuration for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_PROMPT = (
    "VERIFICATION: Before confirming completion, review your work:\n\n"
    "1. Re-read the original request and compare with what you delivered\n"
    "2. Check any files you created or modified — verify content is correct\n"
    "3. If you performed calculations or data transformations, spot-check the results\n"
    "4. Confirm nothing was missed from the original request\n\n"
    "Use read-only tools (ReadFile, ListDirectory, etc.) to verify.\n"
    "If everything is correct, confirm you are done.\n"
    "If you find issues, fix them before completing."
)

_CUSTOM_PROMPT_TEMPLATE = (
    "VERIFICATION: Before confirming completion, review your work:\n\n"
    "1. Re-read the original request and compare with what you delivered\n"
    "2. Verify: {instructions}\n"
    "3. Confirm nothing was missed from the original request\n\n"
    "Use read-only tools (ReadFile, ListDirectory, etc.) to verify.\n"
    "If everything is correct, confirm you are done.\n"
    "If you find issues, fix them before completing."
)


@dataclass
class VerificationConfig:
    """Configuration for the post-completion verification phase."""

    enabled: bool = True
    max_verify_steps: int = 3
    custom_instructions: str = ""

    def build_prompt(self) -> str:
        """Build the verification prompt to inject after agent signals completion."""
        if self.custom_instructions:
            return _CUSTOM_PROMPT_TEMPLATE.format(instructions=self.custom_instructions)
        return _DEFAULT_PROMPT
