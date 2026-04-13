"""M39 Tests: Structured Generation + Tool Use — JSON mode, tool parsing, stop conditions."""

from __future__ import annotations

import json

# ─── Test JSON Constraint ───


class TestJSONConstraint:
    """Tests for JSON validation and repair."""

    def test_valid_json(self):
        """Valid JSON passes through unchanged."""
        from mlx_forge.inference.constrained import JSONConstraint

        c = JSONConstraint()
        text = '{"name": "Alice", "age": 30}'
        result = c.validate_and_repair(text)
        assert json.loads(result) == {"name": "Alice", "age": 30}

    def test_repair_missing_brace(self):
        """Repair missing closing brace."""
        from mlx_forge.inference.constrained import JSONConstraint

        c = JSONConstraint()
        text = '{"name": "Alice"'
        result = c.validate_and_repair(text)
        parsed = json.loads(result)
        assert parsed["name"] == "Alice"

    def test_repair_trailing_comma(self):
        """Repair trailing comma before closing brace."""
        from mlx_forge.inference.constrained import JSONConstraint

        c = JSONConstraint()
        text = '{"a": 1, "b": 2,}'
        result = c.validate_and_repair(text)
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_nested(self):
        """Valid nested JSON works."""
        from mlx_forge.inference.constrained import JSONConstraint

        c = JSONConstraint()
        text = '{"user": {"name": "Bob", "scores": [1, 2, 3]}}'
        result = c.validate_and_repair(text)
        parsed = json.loads(result)
        assert parsed["user"]["name"] == "Bob"
        assert parsed["user"]["scores"] == [1, 2, 3]

    def test_empty_object(self):
        """Empty JSON object."""
        from mlx_forge.inference.constrained import JSONConstraint

        c = JSONConstraint()
        result = c.validate_and_repair("{}")
        assert json.loads(result) == {}

    def test_array(self):
        """JSON array."""
        from mlx_forge.inference.constrained import JSONConstraint

        c = JSONConstraint()
        result = c.validate_and_repair("[1, 2, 3]")
        assert json.loads(result) == [1, 2, 3]

    def test_extract_from_surrounding_text(self):
        """Extract JSON from surrounding text."""
        from mlx_forge.inference.constrained import JSONConstraint

        c = JSONConstraint()
        text = 'Here is the result: {"key": "value"} and more text'
        result = c.validate_and_repair(text)
        parsed = json.loads(result)
        assert parsed["key"] == "value"


# ─── Test JSON Schema ───


class TestJSONSchema:
    """Tests for JSON schema validation."""

    def test_pass(self):
        """Valid JSON against schema passes."""
        from mlx_forge.inference.constrained import JSONSchemaConstraint

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        c = JSONSchemaConstraint(schema)
        ok, err = c.validate_against_schema('{"name": "Alice"}')
        assert ok is True
        assert err is None

    def test_fail_missing_required(self):
        """Missing required field fails."""
        from mlx_forge.inference.constrained import JSONSchemaConstraint

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        c = JSONSchemaConstraint(schema)
        ok, err = c.validate_against_schema('{"age": 30}')
        assert ok is False
        assert "name" in err

    def test_required_fields(self):
        """Multiple required fields checked."""
        from mlx_forge.inference.constrained import JSONSchemaConstraint

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        c = JSONSchemaConstraint(schema)
        ok, _ = c.validate_against_schema('{"name": "Bob", "age": 25}')
        assert ok is True

        ok, err = c.validate_against_schema('{"name": "Bob"}')
        assert ok is False

    def test_type_check(self):
        """Type mismatch is caught."""
        from mlx_forge.inference.constrained import JSONSchemaConstraint

        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        c = JSONSchemaConstraint(schema)
        ok, err = c.validate_against_schema('{"count": "not_a_number"}')
        assert ok is False
        assert "integer" in err

    def test_nested_schema(self):
        """Nested object schema validation."""
        from mlx_forge.inference.constrained import JSONSchemaConstraint

        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
            "required": ["user"],
        }
        c = JSONSchemaConstraint(schema)
        ok, _ = c.validate_against_schema('{"user": {"name": "Alice"}}')
        assert ok is True

        ok, err = c.validate_against_schema('{"user": {"age": 30}}')
        assert ok is False


# ─── Test Tool Parser ───


class TestToolParser:
    """Tests for tool call parsing from model output."""

    def test_parse_xml_tags(self):
        """Parse tool calls in XML format."""
        from mlx_forge.serving.tool_parser import ToolCallParser

        parser = ToolCallParser()
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
        calls = parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "NYC"}

    def test_parse_json_block(self):
        """Parse tool calls in JSON code block."""
        from mlx_forge.serving.tool_parser import ToolCallParser

        parser = ToolCallParser()
        text = '```json\n{"name": "search", "arguments": {"query": "hello"}}\n```'
        calls = parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "search"

    def test_no_calls(self):
        """No tool calls found returns empty list."""
        from mlx_forge.serving.tool_parser import ToolCallParser

        parser = ToolCallParser()
        calls = parser.parse("Just a normal response with no tool calls.")
        assert calls == []

    def test_multiple(self):
        """Multiple tool calls in one response."""
        from mlx_forge.serving.tool_parser import ToolCallParser

        parser = ToolCallParser()
        text = (
            '<tool_call>{"name": "fn1", "arguments": {}}</tool_call>\n'
            '<tool_call>{"name": "fn2", "arguments": {"x": 1}}</tool_call>'
        )
        calls = parser.parse(text)
        assert len(calls) == 2
        assert calls[0].name == "fn1"
        assert calls[1].name == "fn2"

    def test_malformed(self):
        """Malformed tool call JSON is skipped."""
        from mlx_forge.serving.tool_parser import ToolCallParser

        parser = ToolCallParser()
        text = "<tool_call>not json at all</tool_call>"
        calls = parser.parse(text)
        assert calls == []

    def test_format_prompt(self):
        """Tool definitions formatted for prompt injection."""
        from mlx_forge.serving.tool_parser import ToolCallParser

        parser = ToolCallParser()
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "properties": {"city": {"type": "string", "description": "City name"}},
                        "required": ["city"],
                    },
                }
            }
        ]
        result = parser.format_tools_for_prompt(tools)
        assert "get_weather" in result
        assert "city" in result
        assert "tool_call" in result


# ─── Test Stop Checker ───


class TestStopChecker:
    """Tests for unified stop condition checking."""

    def test_token_ids(self):
        """Stop on specific token IDs."""
        from mlx_forge.inference.stop_conditions import StopChecker

        checker = StopChecker(stop_token_ids=[50, 100])
        assert checker.check_token(50) is True
        assert checker.check_token(100) is True
        assert checker.check_token(42) is False

    def test_strings(self):
        """Stop on string sequences."""
        from mlx_forge.inference.stop_conditions import StopChecker

        checker = StopChecker(stop_strings=["<|end|>", "STOP"])
        stopped, text = checker.check_text("Hello world")
        assert stopped is False
        assert text == "Hello world"

        stopped, text = checker.check_text("Hello<|end|>more")
        assert stopped is True
        assert text == "Hello"

    def test_eos(self):
        """EOS token triggers stop."""
        from mlx_forge.inference.stop_conditions import StopChecker

        checker = StopChecker(eos_token_id=2)
        assert checker.check_token(2) is True
        assert checker.check_token(1) is False

    def test_combined(self):
        """All stop conditions work together."""
        from mlx_forge.inference.stop_conditions import StopChecker

        checker = StopChecker(
            stop_strings=["END"],
            stop_token_ids=[99],
            eos_token_id=2,
        )
        assert checker.check_token(2) is True  # EOS
        assert checker.check_token(99) is True  # stop token
        assert checker.check_token(42) is False  # normal token

        stopped, _ = checker.check_text("hello END world")
        assert stopped is True

    def test_no_conditions(self):
        """Empty checker never stops."""
        from mlx_forge.inference.stop_conditions import StopChecker

        checker = StopChecker()
        assert checker.check_token(42) is False
        stopped, text = checker.check_text("anything")
        assert stopped is False

    def test_text_trimming(self):
        """Stop string triggers text trimming at the stop position."""
        from mlx_forge.inference.stop_conditions import StopChecker

        checker = StopChecker(stop_strings=["###"])
        stopped, text = checker.check_text("Here is output### more stuff")
        assert stopped is True
        assert text == "Here is output"


# ─── Test API Types ───


class TestAPITypes:
    """Tests for M39 API type extensions."""

    def test_tool_def(self):
        """ToolDef Pydantic model works."""
        from mlx_forge.serving.openai_types import FunctionDef, ToolDef

        tool = ToolDef(
            function=FunctionDef(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        )
        assert tool.function.name == "get_weather"
        assert tool.type == "function"

    def test_message_with_tool_calls(self):
        """ChatMessage can include tool_calls."""
        from mlx_forge.serving.openai_types import (
            ChatMessage,
            ToolCallFunction,
            ToolCallMessage,
        )

        msg = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCallMessage(
                    function=ToolCallFunction(
                        name="search",
                        arguments='{"q": "test"}',
                    )
                )
            ],
        )
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "search"

    def test_response_format_json(self):
        """ChatCompletionRequest accepts response_format."""
        from mlx_forge.serving.openai_types import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test",
            messages=[],
            response_format={"type": "json_object"},
        )
        assert req.response_format["type"] == "json_object"

    def test_response_format_schema(self):
        """ChatCompletionRequest accepts json_schema response_format."""
        from mlx_forge.serving.openai_types import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test",
            messages=[],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            },
        )
        assert req.response_format["type"] == "json_schema"


# ─── Test Route Integration ───


class TestRouteIntegration:
    """Tests for route-level integration of M39 features."""

    def test_stop_checker_builder(self):
        """_build_stop_checker creates checker from request."""
        from mlx_forge.serving.openai_types import ChatCompletionRequest
        from mlx_forge.serving.routes import _build_stop_checker

        req = ChatCompletionRequest(
            model="test",
            messages=[],
            stop=["STOP"],
            stop_token_ids=[99],
        )
        tokenizer = type("T", (), {"eos_token_id": 2})()
        checker = _build_stop_checker(req, tokenizer)
        assert checker.check_token(2) is True
        assert checker.check_token(99) is True

    def test_json_constraint_applied(self):
        """_apply_json_constraint repairs JSON."""
        from mlx_forge.serving.routes import _apply_json_constraint

        text = '{"key": "value",'
        result = _apply_json_constraint(text, {"type": "json_object"})
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_tool_injection(self):
        """Tools are injected into system prompt."""
        from mlx_forge.serving.openai_types import (
            ChatCompletionRequest,
            FunctionDef,
            ToolDef,
        )
        from mlx_forge.serving.routes import _inject_tools_into_messages

        req = ChatCompletionRequest(
            model="test",
            messages=[],
            tools=[
                ToolDef(function=FunctionDef(name="search", description="Search the web")),
            ],
        )
        messages = [{"role": "user", "content": "hello"}]
        result = _inject_tools_into_messages(messages, req)
        assert len(result) == 2  # system + user
        assert "search" in result[0]["content"]

    def test_tool_parse_from_text(self):
        """Tool calls are parsed from generated text."""
        from mlx_forge.serving.routes import _parse_tool_calls_from_text

        text = '<tool_call>{"name": "calc", "arguments": {"x": 5}}</tool_call>'
        calls = _parse_tool_calls_from_text(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].function.name == "calc"

    def test_no_tool_calls(self):
        """Normal text returns None for tool calls."""
        from mlx_forge.serving.routes import _parse_tool_calls_from_text

        calls = _parse_tool_calls_from_text("Just a normal response.")
        assert calls is None
