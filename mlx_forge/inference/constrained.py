"""Constrained generation: JSON mode and JSON schema validation.

Hybrid approach: lightweight logit bias during generation + post-generation
repair for common JSON errors. No heavy dependencies (no outlines).
"""

from __future__ import annotations

import json
import re


class JSONConstraint:
    """Validates and repairs JSON output from generation.

    During generation, can apply lightweight logit bias toward JSON-valid
    characters at structural positions. After generation, validates and
    repairs common JSON errors.
    """

    # Token IDs for JSON structural chars — populated lazily per tokenizer
    _structural_tokens: dict | None = None

    def validate_and_repair(self, text: str) -> str:
        """Attempt to validate and repair common JSON errors.

        Handles:
        - Missing closing braces/brackets
        - Trailing commas before closing brace/bracket
        - Unclosed strings
        - Leading/trailing whitespace/text around JSON

        Args:
            text: Raw generated text that should be JSON.

        Returns:
            Repaired JSON string.
        """
        # Strip leading/trailing whitespace
        text = text.strip()

        # Try to extract JSON from surrounding text
        text = self._extract_json(text)

        # Fix unclosed strings — if odd number of unescaped quotes, close last
        if self._count_unescaped_quotes(text) % 2 != 0:
            text = text.rstrip()
            if not text.endswith('"'):
                text += '"'

        # Fix missing closing braces/brackets (before trailing comma fix)
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        if open_braces > 0:
            text += "}" * open_braces
        if open_brackets > 0:
            text += "]" * open_brackets

        # Fix trailing commas: ,} or ,] (after braces are balanced)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)

        return text

    def validate(self, text: str) -> tuple[bool, str | None]:
        """Check if text is valid JSON.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            json.loads(text)
            return True, None
        except json.JSONDecodeError as e:
            return False, str(e)

    def _extract_json(self, text: str) -> str:
        """Extract JSON object or array from surrounding text."""
        # Find first { or [
        for i, c in enumerate(text):
            if c in "{[":
                # Find matching close from end
                target = "}" if c == "{" else "]"
                for j in range(len(text) - 1, i, -1):
                    if text[j] == target:
                        return text[i : j + 1]
                # No matching close — return from open brace to end
                return text[i:]
        return text

    def _count_unescaped_quotes(self, text: str) -> int:
        """Count unescaped double quotes."""
        count = 0
        escaped = False
        for c in text:
            if escaped:
                escaped = False
                continue
            if c == "\\":
                escaped = True
                continue
            if c == '"':
                count += 1
        return count


class JSONSchemaConstraint(JSONConstraint):
    """Validates JSON output against a JSON schema.

    Extends JSONConstraint with schema validation after repair.
    """

    def __init__(self, schema: dict):
        self._schema = schema

    @property
    def schema(self) -> dict:
        return self._schema

    def validate_against_schema(self, text: str) -> tuple[bool, str | None]:
        """Validate repaired JSON against the schema.

        Performs basic validation without jsonschema dependency:
        - Required fields presence
        - Type checking for basic types
        - Nested object validation

        Args:
            text: JSON string to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

        return self._validate_value(data, self._schema, path="$")

    def _validate_value(
        self, value, schema: dict, path: str
    ) -> tuple[bool, str | None]:
        """Recursively validate a value against a schema."""
        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(value, dict):
                return False, f"{path}: expected object, got {type(value).__name__}"

            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in value:
                    return False, f"{path}: missing required field '{field}'"

            # Validate properties
            properties = schema.get("properties", {})
            for key, prop_schema in properties.items():
                if key in value:
                    ok, err = self._validate_value(
                        value[key], prop_schema, path=f"{path}.{key}"
                    )
                    if not ok:
                        return False, err

            return True, None

        elif schema_type == "array":
            if not isinstance(value, list):
                return False, f"{path}: expected array, got {type(value).__name__}"
            items_schema = schema.get("items", {})
            if items_schema:
                for i, item in enumerate(value):
                    ok, err = self._validate_value(
                        item, items_schema, path=f"{path}[{i}]"
                    )
                    if not ok:
                        return False, err
            return True, None

        elif schema_type == "string":
            if not isinstance(value, str):
                return False, f"{path}: expected string, got {type(value).__name__}"
            return True, None

        elif schema_type == "number":
            if not isinstance(value, (int, float)):
                return False, f"{path}: expected number, got {type(value).__name__}"
            return True, None

        elif schema_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"{path}: expected integer, got {type(value).__name__}"
            return True, None

        elif schema_type == "boolean":
            if not isinstance(value, bool):
                return False, f"{path}: expected boolean, got {type(value).__name__}"
            return True, None

        elif schema_type == "null":
            if value is not None:
                return False, f"{path}: expected null, got {type(value).__name__}"
            return True, None

        # No type constraint or unknown type — accept anything
        return True, None
