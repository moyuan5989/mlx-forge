"""KV cache for autoregressive text generation.

The cache stores key and value tensors from previous tokens so they don't
need to be recomputed during generation. Each transformer layer gets its
own KVCache instance.

Interface contract (used by all architectures):
    cache.offset -> int: number of tokens already cached
    cache.update_and_fetch(keys, values) -> (all_keys, all_values)
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


class KVCache:
    """Key-value cache for a single transformer layer.

    Uses pre-allocated buffers when max_size is provided (O(T) memory),
    falls back to concatenation when max_size is not set (backward compatible).
    """

    def __init__(self, max_size: int = 0):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0
        self._max_size = max_size
        self._allocated: bool = False

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append new keys/values and return the full cached tensors.

        Args:
            keys: New key tensor of shape (B, n_kv_heads, L, head_dim)
            values: New value tensor of shape (B, n_kv_heads, L, head_dim)

        Returns:
            Tuple of (all_keys, all_values) including both cached and new.
        """
        B, H, L, D = keys.shape

        if self._max_size > 0:
            # Pre-allocated path: slice assignment avoids O(T^2) copies
            if not self._allocated:
                self.keys = mx.zeros((B, H, self._max_size, D), dtype=keys.dtype)
                self.values = mx.zeros((B, H, self._max_size, D), dtype=values.dtype)
                self._allocated = True

            end = self.offset + L
            self.keys[:, :, self.offset:end, :] = keys
            self.values[:, :, self.offset:end, :] = values
            self.offset = end

            return self.keys[:, :, :self.offset, :], self.values[:, :, :self.offset, :]
        else:
            # Concatenation fallback (backward compatible)
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                self.keys = mx.concatenate([self.keys, keys], axis=2)
                self.values = mx.concatenate([self.values, values], axis=2)

            self.offset = self.keys.shape[2]
            return self.keys, self.values

    def trim(self, n: int):
        """Remove last n tokens from cache (for speculative decoding rewind)."""
        self.offset = max(0, self.offset - n)

    def reset(self):
        """Clear the cache."""
        self.keys = None
        self.values = None
        self.offset = 0
        self._allocated = False


class RecurrentCache:
    """Cache for recurrent (DeltaNet) layers.

    Stores two slots:
      [0] = conv_state: last (kernel_size - 1) timesteps for causal conv
      [1] = ssm_state: recurrent state matrix (B, num_heads, key_dim, value_dim)
    """

    def __init__(self):
        self.cache: list[Optional[mx.array]] = [None, None]
        self.offset: int = 0

    def __getitem__(self, idx):
        return self.cache[idx]

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    @property
    def conv_state(self) -> Optional[mx.array]:
        return self.cache[0]

    @conv_state.setter
    def conv_state(self, v: Optional[mx.array]):
        self.cache[0] = v

    @property
    def ssm_state(self) -> Optional[mx.array]:
        return self.cache[1]

    @ssm_state.setter
    def ssm_state(self, v: Optional[mx.array]):
        self.cache[1] = v


class ArraysCache:
    """Generic array cache for SSM/recurrent layers.

    Stores an arbitrary number of state arrays per layer.
    Used by Mamba and other non-attention architectures.
    """

    def __init__(self, size: int = 2):
        self.cache: list[Optional[mx.array]] = [None] * size
        self.offset: int = 0

    def __getitem__(self, idx):
        return self.cache[idx]

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def trim(self, n: int):
        """Remove last n steps from cache."""
        self.offset = max(0, self.offset - n)


def make_cache(num_layers: int, *, max_size: int = 0) -> list[KVCache]:
    """Create a list of KV caches, one per transformer layer.

    Args:
        num_layers: Number of transformer layers.
        max_size: Pre-allocate buffer for this many tokens. 0 = dynamic (concat).
    """
    return [KVCache(max_size=max_size) for _ in range(num_layers)]
