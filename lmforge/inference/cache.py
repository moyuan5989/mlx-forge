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

    Stores concatenated keys and values from all previous positions.
    Used during autoregressive generation to avoid recomputing attention
    over the full sequence at each step.
    """

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0

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
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.offset = self.keys.shape[2]
        return self.keys, self.values

    def reset(self):
        """Clear the cache."""
        self.keys = None
        self.values = None
        self.offset = 0


def make_cache(num_layers: int) -> list[KVCache]:
    """Create a list of KV caches, one per transformer layer."""
    return [KVCache() for _ in range(num_layers)]
