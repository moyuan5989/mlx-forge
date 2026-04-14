"""Thinking model support — parse <think> blocks from reasoning models.

Supports DeepSeek-R1, Qwen3 (with enable_thinking), QwQ, and other
models that emit chain-of-thought in structured tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ThinkingResult:
    """Parsed thinking + response from a reasoning model."""

    thinking: str  # chain-of-thought content (may be empty)
    response: str  # final answer


class ThinkingParser:
    """Parse thinking model output, separating reasoning from response.

    Detects and extracts content within <think>...</think> or
    <thinking>...</thinking> blocks. Multiple blocks are concatenated.
    """

    _PATTERNS = [
        re.compile(r"<think>(.*?)</think>", re.DOTALL),
        re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL),
    ]

    def parse(self, text: str) -> ThinkingResult:
        """Parse thinking blocks from generated text.

        Args:
            text: Full model output that may contain thinking tags.

        Returns:
            ThinkingResult with separated thinking and response.
        """
        thinking_parts = []
        remaining = text

        for pattern in self._PATTERNS:
            matches = pattern.findall(remaining)
            if matches:
                thinking_parts.extend(m.strip() for m in matches)
                remaining = pattern.sub("", remaining)

        thinking = "\n\n".join(thinking_parts)
        response = remaining.strip()

        return ThinkingResult(thinking=thinking, response=response)

    def has_thinking_tags(self, text: str) -> bool:
        """Check if text contains any thinking tags."""
        for pattern in self._PATTERNS:
            if pattern.search(text):
                return True
        return False
