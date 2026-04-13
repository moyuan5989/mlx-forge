"""Rotating KV cache for context window overflow handling.

When the cache fills beyond max_size, keeps the first num_keep tokens
(system prompt / prefix) and the most recent tokens, discarding the
middle. This matches Ollama's behavior with num_keep + sliding window.
"""

from __future__ import annotations

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


class RotatingKVCache:
    """Fixed-size KV cache with sliding window eviction.

    Same interface as KVCache (offset, update_and_fetch, trim, reset)
    but handles overflow by rotating: keeps first num_keep tokens and
    the most recent tokens that fit.

    Args:
        max_size: Maximum number of tokens the cache can hold.
        num_keep: Number of tokens at the start to always preserve
            (e.g., system prompt tokens). 0 = no preserved prefix.
    """

    def __init__(self, max_size: int, num_keep: int = 0):
        if max_size <= 0:
            raise ValueError("max_size must be positive for RotatingKVCache")
        if num_keep < 0:
            raise ValueError("num_keep must be non-negative")
        if num_keep >= max_size:
            raise ValueError("num_keep must be less than max_size")

        self.max_size = max_size
        self.num_keep = num_keep
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0
        self._rotated: bool = False  # whether rotation has occurred

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append new keys/values, rotating if necessary.

        Args:
            keys: New key tensor of shape (B, n_kv_heads, L, head_dim)
            values: New value tensor of shape (B, n_kv_heads, L, head_dim)

        Returns:
            Tuple of (all_keys, all_values) for the current window.
        """
        B, H, L, D = keys.shape

        if self.keys is None:
            # First call — allocate buffer
            self.keys = mx.zeros((B, H, self.max_size, D), dtype=keys.dtype)
            self.values = mx.zeros((B, H, self.max_size, D), dtype=values.dtype)

        new_end = self.offset + L

        if new_end <= self.max_size:
            # Fits without rotation
            self.keys[:, :, self.offset:new_end, :] = keys
            self.values[:, :, self.offset:new_end, :] = values
            self.offset = new_end
        else:
            # Need rotation — keep first num_keep + most recent
            self._rotate(keys, values, L)

        return (
            self.keys[:, :, : self.offset, :],
            self.values[:, :, : self.offset, :],
        )

    def _rotate(self, new_keys: mx.array, new_values: mx.array, new_len: int):
        """Rotate cache: keep prefix + most recent tokens + new tokens."""
        if not self._rotated:
            logger.warning(
                "Context window full (%d tokens). Rotating cache "
                "(keeping %d prefix tokens). Some middle context will be lost.",
                self.offset,
                self.num_keep,
            )
            self._rotated = True

        # How much space do we need for new tokens?
        available = self.max_size - self.num_keep
        # Keep as many recent tokens as possible after adding new ones
        recent_budget = available - new_len
        if recent_budget < 0:
            # New tokens alone exceed available space — just keep what fits
            recent_budget = 0
            new_len = min(new_len, available)

        # Gather the pieces:
        # 1. Prefix: keys[:, :, :num_keep, :]  (already in place)
        # 2. Recent: keys[:, :, offset-recent_budget:offset, :]
        # 3. New: new_keys[:, :, :new_len, :]

        if recent_budget > 0 and self.offset > self.num_keep:
            recent_start = max(self.num_keep, self.offset - recent_budget)
            actual_recent = self.offset - recent_start
            # Move recent tokens right after prefix
            dst_start = self.num_keep
            dst_end = dst_start + actual_recent
            self.keys[:, :, dst_start:dst_end, :] = (
                self.keys[:, :, recent_start : recent_start + actual_recent, :]
            )
            self.values[:, :, dst_start:dst_end, :] = (
                self.values[:, :, recent_start : recent_start + actual_recent, :]
            )
            new_offset = dst_end
        else:
            new_offset = self.num_keep

        # Place new tokens
        end = min(new_offset + new_len, self.max_size)
        actual_new = end - new_offset
        self.keys[:, :, new_offset:end, :] = new_keys[:, :, :actual_new, :]
        self.values[:, :, new_offset:end, :] = new_values[:, :, :actual_new, :]
        self.offset = end

    def trim(self, n: int):
        """Remove last n tokens from cache."""
        self.offset = max(0, self.offset - n)

    def reset(self):
        """Clear the cache."""
        self.keys = None
        self.values = None
        self.offset = 0
        self._rotated = False

    @property
    def is_full(self) -> bool:
        """Whether the cache has reached max_size."""
        return self.offset >= self.max_size

    @property
    def has_rotated(self) -> bool:
        """Whether rotation has occurred (context was truncated)."""
        return self._rotated


def make_rotating_cache(
    num_layers: int, *, max_size: int, num_keep: int = 0
) -> list[RotatingKVCache]:
    """Create rotating caches for context overflow handling.

    Args:
        num_layers: Number of transformer layers.
        max_size: Maximum tokens per cache.
        num_keep: Tokens to always preserve at start.
    """
    return [RotatingKVCache(max_size=max_size, num_keep=num_keep) for _ in range(num_layers)]
