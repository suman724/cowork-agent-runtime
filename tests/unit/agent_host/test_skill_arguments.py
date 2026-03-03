"""Tests for $ARGUMENTS substitution in skill content."""

from __future__ import annotations

from agent_host.skills.skill_loader import substitute_arguments


class TestSubstituteArguments:
    def test_arguments_all_values(self) -> None:
        """$ARGUMENTS should be replaced with all values joined by space."""
        content = "Search for $ARGUMENTS in the codebase."
        result = substitute_arguments(content, {"query": "auth", "scope": "src"})
        assert result == "Search for auth src in the codebase."

    def test_arguments_indexed_zero(self) -> None:
        """$ARGUMENTS[0] should be replaced with the first argument value."""
        content = "Deploy to $ARGUMENTS[0] environment."
        result = substitute_arguments(content, {"target": "staging"})
        assert result == "Deploy to staging environment."

    def test_arguments_indexed_multiple(self) -> None:
        """$ARGUMENTS[0] and $ARGUMENTS[1] should resolve to positional values."""
        content = "Copy $ARGUMENTS[0] to $ARGUMENTS[1]."
        result = substitute_arguments(content, {"src": "/a/b", "dest": "/c/d"})
        assert result == "Copy /a/b to /c/d."

    def test_arguments_out_of_range(self) -> None:
        """Out-of-range index should resolve to empty string."""
        content = "Value: $ARGUMENTS[5]."
        result = substitute_arguments(content, {"only": "one"})
        assert result == "Value: ."

    def test_no_arguments_empty_replacement(self) -> None:
        """No arguments should replace placeholders with empty strings."""
        content = "Search for $ARGUMENTS in $ARGUMENTS[0] dir."
        result = substitute_arguments(content, {})
        assert result == "Search for  in  dir."

    def test_mixed_placeholders(self) -> None:
        """Mixed $ARGUMENTS and $ARGUMENTS[N] in same content."""
        content = "All: $ARGUMENTS, first: $ARGUMENTS[0], second: $ARGUMENTS[1]."
        result = substitute_arguments(content, {"a": "foo", "b": "bar"})
        assert result == "All: foo bar, first: foo, second: bar."

    def test_no_placeholders_unchanged(self) -> None:
        """Content without placeholders should be returned unchanged."""
        content = "Just plain text with no placeholders."
        result = substitute_arguments(content, {"key": "value"})
        assert result == "Just plain text with no placeholders."

    def test_empty_content(self) -> None:
        """Empty content should return empty string."""
        result = substitute_arguments("", {"key": "value"})
        assert result == ""

    def test_single_argument_value(self) -> None:
        """Single argument with $ARGUMENTS."""
        content = "Run: $ARGUMENTS"
        result = substitute_arguments(content, {"cmd": "make test"})
        assert result == "Run: make test"

    def test_numeric_argument_values(self) -> None:
        """Numeric argument values should be converted to strings."""
        content = "Port: $ARGUMENTS[0], count: $ARGUMENTS[1]"
        result = substitute_arguments(content, {"port": 8080, "count": 3})
        assert result == "Port: 8080, count: 3"
