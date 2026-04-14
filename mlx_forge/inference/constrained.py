"""Constrained generation: JSON mode and JSON schema validation.

Hybrid approach: lightweight logit bias during generation + post-generation
repair for common JSON errors. No heavy dependencies (no outlines).
"""

from __future__ import annotations

import json
import re
from typing import Callable

import mlx.core as mx
import numpy as np


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


class JSONLogitProcessor:
    """Lightweight FSM that biases logits toward valid JSON during generation.

    Tracks JSON structural state (inside string, nesting depth, expected
    next token class) and applies logit bias to suppress tokens that would
    produce invalid JSON structure.

    Not as robust as outlines, but zero-dependency and handles most cases.

    Usage:
        processor = JSONLogitProcessor(tokenizer)
        # In generation loop:
        logits = processor.process(logits, generated_text_so_far)
    """

    # Characters that are always valid in JSON strings
    _BIAS_STRENGTH = 10.0  # logit boost for structural tokens

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._token_char_map: dict[int, str] | None = None
        self._structural_ids: dict[str, list[int]] | None = None

    def process(self, logits: mx.array, generated_text: str) -> mx.array:
        """Apply logit bias based on current JSON state.

        Args:
            logits: Raw logits of shape (vocab_size,).
            generated_text: Text generated so far.

        Returns:
            Modified logits with structural bias applied.
        """
        if self._structural_ids is None:
            self._build_token_map()

        state = self._analyze_state(generated_text)

        if state == "EXPECT_OPEN":
            # Must start with { or [
            return self._boost_tokens(logits, ["{", "["])
        elif state == "EXPECT_KEY_OR_CLOSE":
            # Inside object: expect " (for key) or }
            return self._boost_tokens(logits, ['"', "}"])
        elif state == "EXPECT_COLON":
            return self._boost_tokens(logits, [":"])
        elif state == "EXPECT_VALUE":
            # Expect: { [ " digit true false null
            return self._boost_tokens(logits, ['"', "{", "[", "t", "f", "n", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        elif state == "EXPECT_COMMA_OR_CLOSE":
            return self._boost_tokens(logits, [",", "}", "]"])
        elif state == "IN_STRING":
            # Inside string — suppress only raw newlines (allow everything else)
            return self._suppress_tokens(logits, ["\n"])

        return logits

    def _analyze_state(self, text: str) -> str:
        """Determine current JSON parser state from generated text."""
        text = text.strip()
        if not text:
            return "EXPECT_OPEN"

        in_string = False
        escape_next = False
        depth = 0
        last_structural = ""

        for c in text:
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                if in_string:
                    escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                if not in_string:
                    last_structural = "STRING_END"
                else:
                    last_structural = "STRING_START"
                continue
            if in_string:
                continue

            if c in "{[":
                depth += 1
                last_structural = "OPEN"
            elif c in "}]":
                depth -= 1
                last_structural = "CLOSE"
            elif c == ":":
                last_structural = "COLON"
            elif c == ",":
                last_structural = "COMMA"

        if in_string:
            return "IN_STRING"

        if depth <= 0 and last_structural in ("CLOSE", ""):
            if not text:
                return "EXPECT_OPEN"
            return "DONE"

        if last_structural == "OPEN":
            return "EXPECT_KEY_OR_CLOSE"
        elif last_structural == "COMMA":
            # After comma in object → expect key; in array → expect value
            # Simplified: we allow both
            return "EXPECT_VALUE"
        elif last_structural == "STRING_END":
            # After a string: could be key (expect colon) or value (expect comma/close)
            # Look back to determine context
            stripped = text.rstrip()
            colon_pos = stripped.rfind(":")
            comma_pos = max(stripped.rfind(","), stripped.rfind("{"), stripped.rfind("["))
            if colon_pos > comma_pos:
                # We're after a value
                return "EXPECT_COMMA_OR_CLOSE"
            else:
                # We're after a key
                return "EXPECT_COLON"
        elif last_structural == "COLON":
            return "EXPECT_VALUE"
        elif last_structural == "CLOSE":
            return "EXPECT_COMMA_OR_CLOSE"

        return "EXPECT_VALUE"

    def _boost_tokens(self, logits: mx.array, chars: list[str]) -> mx.array:
        """Boost logits for tokens that start with any of the given characters."""
        if not self._structural_ids:
            return logits

        boost_ids = set()
        for c in chars:
            if c in self._structural_ids:
                boost_ids.update(self._structural_ids[c])

        if not boost_ids:
            return logits

        boost_np = np.zeros(logits.shape[-1], dtype=np.float32)
        for tid in boost_ids:
            if 0 <= tid < logits.shape[-1]:
                boost_np[tid] = self._BIAS_STRENGTH

        return logits + mx.array(boost_np)

    def _suppress_tokens(self, logits: mx.array, chars: list[str]) -> mx.array:
        """Suppress logits for tokens matching given characters."""
        if not self._structural_ids:
            return logits

        suppress_ids = set()
        for c in chars:
            if c in self._structural_ids:
                suppress_ids.update(self._structural_ids[c])

        if not suppress_ids:
            return logits

        suppress_np = np.zeros(logits.shape[-1], dtype=np.float32)
        for tid in suppress_ids:
            if 0 <= tid < logits.shape[-1]:
                suppress_np[tid] = -self._BIAS_STRENGTH

        return logits + mx.array(suppress_np)

    def _build_token_map(self) -> None:
        """Build mapping from characters to token IDs."""
        self._structural_ids = {}

        # Structural characters we want to map
        chars_to_map = list('{}[]":,tfn-0123456789.\n ')

        vocab_size = getattr(self._tokenizer, "vocab_size", 32000)
        for tid in range(min(vocab_size, 128256)):
            try:
                decoded = self._tokenizer.decode([tid])
                if decoded:
                    first_char = decoded[0] if len(decoded) > 0 else ""
                    stripped = decoded.strip()
                    # Map by first character for structural tokens
                    if first_char in chars_to_map:
                        if first_char not in self._structural_ids:
                            self._structural_ids[first_char] = []
                        self._structural_ids[first_char].append(tid)
                    # Also map exact single-char tokens
                    if stripped in chars_to_map and len(stripped) == 1:
                        if stripped not in self._structural_ids:
                            self._structural_ids[stripped] = []
                        if tid not in self._structural_ids[stripped]:
                            self._structural_ids[stripped].append(tid)
            except Exception:
                continue


def make_json_logit_processor(
    tokenizer, response_format: dict | None
) -> Callable[[mx.array, str], mx.array] | None:
    """Create a JSON logit processor if response_format requests it.

    Args:
        tokenizer: Tokenizer for token mapping.
        response_format: Request's response_format dict.

    Returns:
        A callable(logits, text) -> logits, or None if not needed.
    """
    if not response_format:
        return None

    fmt_type = response_format.get("type")
    if fmt_type not in ("json_object", "json_schema"):
        return None

    processor = JSONLogitProcessor(tokenizer)
    return processor.process
