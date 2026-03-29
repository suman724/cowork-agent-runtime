"""Tests for SensitiveDetector — element sensitivity and form data redaction."""

from __future__ import annotations

from tool_runtime.tools.browser.dom_service import IndexedElement
from tool_runtime.tools.browser.sensitive_detector import (
    SensitiveType,
    detect_sensitive,
    redact_form_value,
)


def _make_element(
    role: str = "textbox",
    name: str = "Field",
    raw: dict | None = None,
    **kwargs: object,
) -> IndexedElement:
    return IndexedElement(
        index=1,
        role=role,
        name=name,
        raw=raw or {},
        **kwargs,  # type: ignore[arg-type]
    )


class TestDetectSensitive:
    def test_password_field_by_input_type(self) -> None:
        el = _make_element(raw={"inputType": "password"})
        assert detect_sensitive(el) == SensitiveType.PASSWORD

    def test_password_field_by_name(self) -> None:
        el = _make_element(name="Enter password")
        assert detect_sensitive(el) == SensitiveType.PASSWORD

    def test_payment_field_by_autocomplete(self) -> None:
        el = _make_element(raw={"autocomplete": "cc-number"})
        assert detect_sensitive(el) == SensitiveType.PAYMENT

    def test_payment_field_by_name(self) -> None:
        el = _make_element(name="Credit card number")
        assert detect_sensitive(el) == SensitiveType.PAYMENT

    def test_pii_ssn_field(self) -> None:
        el = _make_element(name="SSN")
        assert detect_sensitive(el) == SensitiveType.PII

    def test_pii_tax_id_field(self) -> None:
        el = _make_element(name="Tax ID", description="Enter your tax identifier")
        assert detect_sensitive(el) == SensitiveType.PII

    def test_destructive_delete_button(self) -> None:
        el = _make_element(role="button", name="Delete project")
        assert detect_sensitive(el) == SensitiveType.DESTRUCTIVE

    def test_destructive_remove_button(self) -> None:
        el = _make_element(role="button", name="Remove from cart")
        assert detect_sensitive(el) == SensitiveType.DESTRUCTIVE

    def test_destructive_cancel_subscription(self) -> None:
        el = _make_element(role="button", name="Cancel subscription")
        assert detect_sensitive(el) == SensitiveType.DESTRUCTIVE

    def test_destructive_deactivate(self) -> None:
        el = _make_element(role="button", name="Deactivate account")
        assert detect_sensitive(el) == SensitiveType.DESTRUCTIVE

    def test_normal_text_field_not_flagged(self) -> None:
        el = _make_element(name="First name")
        assert detect_sensitive(el) is None

    def test_normal_button_not_flagged(self) -> None:
        el = _make_element(role="button", name="Save changes")
        assert detect_sensitive(el) is None

    def test_normal_link_not_flagged(self) -> None:
        el = _make_element(role="link", name="Go to dashboard")
        assert detect_sensitive(el) is None

    def test_submit_button_not_destructive(self) -> None:
        """Submit is not destructive — it's handled by Tier 3 (BrowserSubmit)."""
        el = _make_element(role="button", name="Submit form")
        assert detect_sensitive(el) is None


class TestRedactFormValue:
    def test_password_full_redact(self) -> None:
        assert redact_form_value("password", "", "pw", "secretpass") == "••••••"

    def test_cc_number_last_4(self) -> None:
        result = redact_form_value("text", "cc-number", "card", "4111111111111234")
        assert result == "••••••••1234"

    def test_cc_csc_full_redact(self) -> None:
        assert redact_form_value("text", "", "cvv", "123") == "•••"

    def test_cc_exp_not_redacted(self) -> None:
        assert redact_form_value("text", "", "expiry", "12/28") == "12/28"

    def test_ssn_last_4(self) -> None:
        result = redact_form_value("text", "", "ssn", "123-45-6789")
        assert result == "•••-••-6789"

    def test_short_value_fully_redacted(self) -> None:
        assert redact_form_value("text", "", "state", "CA") == "••"

    def test_normal_field_not_redacted(self) -> None:
        assert redact_form_value("text", "", "name", "John Doe") == "John Doe"

    def test_empty_value_unchanged(self) -> None:
        assert redact_form_value("text", "", "field", "") == ""

    def test_single_char_fully_redacted(self) -> None:
        assert redact_form_value("text", "", "field", "A") == "••"
