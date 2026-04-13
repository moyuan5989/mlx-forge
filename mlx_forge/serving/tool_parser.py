"""Tool/function call parsing from model output.

Supports multiple formats:
- <tool_call> XML tags (Qwen, Hermes style)
- JSON code blocks with function structure
- Generic {"name": ..., "arguments": ...} JSON
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ToolCall:
    """A parsed function/tool call."""

    name: str
    arguments: dict


class ToolCallParser:
    """Parses tool calls from model-generated text.

    Supports multiple common formats that models use for tool calling.
    """

    # Patterns for different tool call formats
    _XML_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
    )
    _JSON_BLOCK_PATTERN = re.compile(
        r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL
    )

    def parse(self, text: str) -> list[ToolCall]:
        """Parse tool calls from generated text.

        Tries each format in order:
        1. <tool_call> XML tags
        2. JSON code blocks
        3. Inline JSON with name/arguments structure

        Args:
            text: Model-generated text.

        Returns:
            List of parsed ToolCall objects (may be empty).
        """
        calls = []

        # 1. Try XML format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        for match in self._XML_PATTERN.finditer(text):
            call = self._parse_json_call(match.group(1))
            if call:
                calls.append(call)

        if calls:
            return calls

        # 2. Try JSON code blocks: ```json {"name": ..., "arguments": ...} ```
        for match in self._JSON_BLOCK_PATTERN.finditer(text):
            call = self._parse_json_call(match.group(1))
            if call:
                calls.append(call)

        if calls:
            return calls

        # 3. Try inline JSON — find any {"name": ..., "arguments": ...} pattern
        calls = self._parse_inline_json(text)

        return calls

    def format_tools_for_prompt(self, tools: list[dict]) -> str:
        """Format tool definitions for injection into the system prompt.

        Generates a clear text description of available tools that models
        can understand for function calling.

        Args:
            tools: List of tool definition dicts (OpenAI format).

        Returns:
            Formatted string describing available tools.
        """
        if not tools:
            return ""

        lines = ["You have access to the following tools:\n"]

        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})

            lines.append(f"### {name}")
            if desc:
                lines.append(f"{desc}")

            properties = params.get("properties", {})
            required = params.get("required", [])
            if properties:
                lines.append("Parameters:")
                for pname, pdef in properties.items():
                    ptype = pdef.get("type", "any")
                    pdesc = pdef.get("description", "")
                    req = " (required)" if pname in required else ""
                    lines.append(f"  - {pname}: {ptype}{req} — {pdesc}")

            lines.append("")

        lines.append(
            "To call a tool, respond with a JSON object in <tool_call> tags:\n"
            '<tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>'
        )

        return "\n".join(lines)

    def _parse_json_call(self, text: str) -> ToolCall | None:
        """Try to parse a JSON string as a tool call."""
        text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        name = data.get("name")
        if not name:
            return None

        arguments = data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}

        return ToolCall(name=str(name), arguments=arguments if isinstance(arguments, dict) else {})

    def _parse_inline_json(self, text: str) -> list[ToolCall]:
        """Find inline JSON objects that look like tool calls."""
        calls = []
        # Find JSON objects in text
        depth = 0
        start = None
        for i, c in enumerate(text):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start : i + 1]
                    call = self._parse_json_call(candidate)
                    if call:
                        calls.append(call)
                    start = None
        return calls
