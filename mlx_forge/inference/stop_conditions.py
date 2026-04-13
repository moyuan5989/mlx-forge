"""Unified stop condition checking for text generation.

Replaces duplicated stop logic across streaming/non-streaming code paths.
"""

from __future__ import annotations


class StopChecker:
    """Check multiple stop conditions during generation.

    Handles:
    - EOS token ID
    - Custom stop token IDs
    - Stop strings (with text trimming)

    Args:
        stop_strings: List of string sequences that trigger a stop.
        stop_token_ids: List of token IDs that trigger a stop.
        eos_token_id: The model's end-of-sequence token ID.
    """

    def __init__(
        self,
        stop_strings: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        eos_token_id: int | None = None,
    ):
        self._stop_strings = stop_strings or []
        self._stop_token_ids = set(stop_token_ids or [])
        self._eos_token_id = eos_token_id

    def check_token(self, token_id: int) -> bool:
        """Check if a token ID should trigger a stop.

        Args:
            token_id: The generated token ID.

        Returns:
            True if generation should stop.
        """
        if self._eos_token_id is not None and token_id == self._eos_token_id:
            return True
        return token_id in self._stop_token_ids

    def check_text(self, text: str) -> tuple[bool, str]:
        """Check if generated text contains a stop string.

        If a stop string is found, the text is trimmed to exclude it.

        Args:
            text: The accumulated generated text so far.

        Returns:
            Tuple of (should_stop, trimmed_text).
        """
        for seq in self._stop_strings:
            idx = text.find(seq)
            if idx != -1:
                return True, text[:idx]
        return False, text

    @property
    def has_stop_strings(self) -> bool:
        """Whether any stop strings are configured."""
        return len(self._stop_strings) > 0

    @property
    def has_stop_tokens(self) -> bool:
        """Whether any custom stop token IDs are configured."""
        return len(self._stop_token_ids) > 0
