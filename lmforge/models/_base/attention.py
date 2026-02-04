"""Attention utilities for model architectures."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
) -> mx.array:
    """
    Create a causal attention mask.

    Args:
        N: Sequence length
        offset: Position offset for KV cache
        window_size: Optional sliding window size

    Returns:
        Boolean mask array of shape (N, offset + N)
    """
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


def create_attention_mask(
    h: mx.array,
    cache=None,
    window_size: Optional[int] = None,
    return_array: bool = False,
):
    """
    Create attention mask for transformer layers.

    Args:
        h: Hidden states tensor of shape (B, L, D)
        cache: Optional KV cache with offset attribute
        window_size: Optional sliding window size
        return_array: If True, always return array instead of "causal" string

    Returns:
        Either "causal" string (for optimized path), None (single token),
        or explicit mask array
    """
    N = h.shape[1]

    # For KV cache with make_mask method
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)

    # Single token doesn't need mask
    if N == 1:
        return None

    # Return explicit mask if requested or sliding window
    if return_array or (window_size is not None and N > window_size):
        return create_causal_mask(N, window_size=window_size)

    # Use optimized causal path
    return "causal"


def scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """
    Compute scaled dot-product attention.

    Uses MLX's fast SDPA implementation for optimal performance.

    Args:
        queries: Query tensor of shape (B, n_heads, L, head_dim)
        keys: Key tensor of shape (B, n_kv_heads, S, head_dim)
        values: Value tensor of shape (B, n_kv_heads, S, head_dim)
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        mask: Optional attention mask

    Returns:
        Attention output of shape (B, n_heads, L, head_dim)
    """
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
    )
