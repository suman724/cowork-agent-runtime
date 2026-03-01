"""Base tool abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from tool_runtime.exceptions import ToolInputValidationError

if TYPE_CHECKING:
    from tool_runtime.models import ExecutionContext, RawToolOutput

_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _validate_property(field: str, value: Any, prop_schema: dict[str, Any], tool_name: str) -> None:
    """Validate a single property value against its JSON Schema definition."""
    expected_type = prop_schema.get("type")
    if expected_type and expected_type in _JSON_TYPE_MAP:
        py_type = _JSON_TYPE_MAP[expected_type]
        # JSON has no boolean/int distinction — bool is a subclass of int in Python,
        # so reject bools when an integer is expected and vice versa.
        if expected_type == "integer" and isinstance(value, bool):
            raise ToolInputValidationError(
                f"Argument '{field}' for tool {tool_name} must be {expected_type}, "
                f"got {type(value).__name__}"
            )
        if not isinstance(value, py_type):
            raise ToolInputValidationError(
                f"Argument '{field}' for tool {tool_name} must be {expected_type}, "
                f"got {type(value).__name__}"
            )

    if (
        "minimum" in prop_schema
        and isinstance(value, (int, float))
        and value < prop_schema["minimum"]
    ):
        raise ToolInputValidationError(
            f"Argument '{field}' for tool {tool_name} must be >= {prop_schema['minimum']}, "
            f"got {value}"
        )

    if (
        "maximum" in prop_schema
        and isinstance(value, (int, float))
        and value > prop_schema["maximum"]
    ):
        raise ToolInputValidationError(
            f"Argument '{field}' for tool {tool_name} must be <= {prop_schema['maximum']}, "
            f"got {value}"
        )

    if "enum" in prop_schema and value not in prop_schema["enum"]:
        raise ToolInputValidationError(
            f"Argument '{field}' for tool {tool_name} must be one of {prop_schema['enum']}, "
            f"got '{value}'"
        )


class BaseTool(ABC):
    """Abstract base class for all built-in tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name (e.g., ReadFile, WriteFile)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description sent to the LLM."""
        ...

    @property
    @abstractmethod
    def capability(self) -> str:
        """Required capability (e.g., File.Read, Shell.Exec)."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema describing the tool's input arguments."""
        ...

    @abstractmethod
    async def execute(
        self,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> RawToolOutput:
        """Execute the tool with the given arguments.

        Raises ToolRuntimeError subclasses on failure.
        """
        ...

    def validate_input(self, arguments: dict[str, Any]) -> None:
        """Validate arguments against the tool's JSON Schema.

        Checks required fields, types, minimum/maximum, enum values,
        and additionalProperties. Subclasses can override for custom validation.
        """
        schema = self.input_schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in arguments:
                raise ToolInputValidationError(
                    f"Missing required argument '{field}' for tool {self.name}"
                )

        if schema.get("additionalProperties") is False:
            extra = set(arguments) - set(properties)
            if extra:
                raise ToolInputValidationError(f"Unknown argument(s) {extra} for tool {self.name}")

        for field, value in arguments.items():
            prop_schema = properties.get(field)
            if prop_schema is None:
                continue
            _validate_property(field, value, prop_schema, self.name)
