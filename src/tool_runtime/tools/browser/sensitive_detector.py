"""SensitiveDetector — heuristic detection of sensitive page elements.

Detects password fields, payment fields, destructive buttons, and PII
fields. Used by Tier 2 HITL to trigger approval before interaction.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tool_runtime.tools.browser.dom_service import IndexedElement


class SensitiveType(Enum):
    """Types of sensitive elements detected."""

    PASSWORD = "password"  # noqa: S105
    PAYMENT = "payment"
    DESTRUCTIVE = "destructive"
    PII = "pii"


# --- Detection patterns ---

_DESTRUCTIVE_TEXT_PATTERNS = re.compile(
    r"\b(delete|remove|cancel\s+subscription|deactivate|"
    r"destroy|revoke|terminate|close\s+account|"
    r"unsubscribe|permanently)\b",
    re.IGNORECASE,
)

_PAYMENT_PATTERNS = re.compile(
    r"(cc[-_]?(number|num|csc|cvv|cvc|exp)|"
    r"card[-_\s]?number|credit[-_\s]?card|"
    r"payment|billing)",
    re.IGNORECASE,
)

_PII_PATTERNS = re.compile(
    r"(ssn|social[-_\s]?security|tax[-_\s]?id|"
    r"national[-_\s]?id|passport[-_\s]?number|"
    r"driver[-_\s]?licen[sc]e)",
    re.IGNORECASE,
)


def detect_sensitive(element: IndexedElement) -> SensitiveType | None:
    """Detect if an element is sensitive and requires approval.

    Returns the type of sensitivity, or None if the element is not sensitive.
    """
    role = element.role
    name = element.name
    raw = element.raw

    # 1. Password fields — highest confidence
    input_type = raw.get("inputType", "")
    autocomplete = raw.get("autocomplete", "")

    if input_type == "password" or (role == "textbox" and "password" in name.lower()):
        return SensitiveType.PASSWORD

    # 2. Payment fields
    if autocomplete and _PAYMENT_PATTERNS.search(autocomplete):
        return SensitiveType.PAYMENT

    name_and_desc = f"{name} {element.description}"
    if _PAYMENT_PATTERNS.search(name_and_desc):
        return SensitiveType.PAYMENT

    # 3. PII fields (SSN, tax ID, etc.)
    if _PII_PATTERNS.search(name_and_desc):
        return SensitiveType.PII

    # 4. Destructive buttons
    if role in ("button", "link", "menuitem") and _DESTRUCTIVE_TEXT_PATTERNS.search(name):
        return SensitiveType.DESTRUCTIVE

    return None


# --- Form data redaction ---

_CC_NUMBER_PATTERN = re.compile(r"cc[-_]?(number|num)", re.IGNORECASE)
_CC_CSC_PATTERN = re.compile(r"(cvv|cvc|csc)", re.IGNORECASE)
_CC_EXP_PATTERN = re.compile(r"expir", re.IGNORECASE)
_SSN_PATTERN = re.compile(r"(ssn|social[-_]?security|tax[-_]?id)", re.IGNORECASE)


def redact_form_value(
    field_type: str,
    autocomplete: str,
    field_name: str,
    value: str,
) -> str:
    """Redact a form field value for the submission approval dialog.

    Rules:
    - password → "••••••"
    - credit card number → last 4 digits ("••••••••1234")
    - CVV/CSC → "•••"
    - expiration → not redacted (low sensitivity)
    - SSN/tax ID → last 4 ("•••-••-1234")
    - short values (≤2 chars) → "••"
    - all other → not redacted
    """
    if not value:
        return value

    # Short values always fully redacted
    if len(value) <= 2:
        return "••"

    label_context = f"{autocomplete} {field_name}"

    # Password
    if field_type == "password":
        return "••••••"

    # Credit card number — show last 4
    if _CC_NUMBER_PATTERN.search(label_context):
        return f"••••••••{value[-4:]}" if len(value) >= 4 else "••••"

    # CVV/CSC — full redaction
    if _CC_CSC_PATTERN.search(label_context):
        return "•••"

    # Expiration — not redacted
    if _CC_EXP_PATTERN.search(label_context):
        return value

    # SSN/tax ID — show last 4
    digits = re.sub(r"\D", "", value)
    if _SSN_PATTERN.search(label_context) and len(digits) >= 4:
        return f"•••-••-{digits[-4:]}"

    return value
