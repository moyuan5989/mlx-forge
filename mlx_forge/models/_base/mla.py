"""Multi-Latent Attention (MLA) for DeepSeek architectures.

Compressed KV via low-rank projection, reducing KV cache by 93.5%.
K = Wk @ compress(x), V = Wv @ compress(x)
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .attention import scaled_dot_product_attention
from .rope import initialize_rope


class MultiLatentAttention(nn.Module):
    """Multi-Latent Attention with compressed KV cache.

    Uses low-rank projections to compress key-value representations,
    dramatically reducing KV cache memory.

    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Per-head dimension
        q_lora_rank: Rank for query compression (0 = no compression)
        kv_lora_rank: Rank for KV compression
        qk_rope_head_dim: Dimension for rotary-embedded QK
        qk_nope_head_dim: Dimension for non-rotary QK
        rope_theta: Base frequency for RoPE
        max_position_embeddings: Maximum sequence length
        rope_scaling: RoPE scaling config
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        rope_scaling: dict = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.scale = head_dim ** -0.5

        # Query projection
        if q_lora_rank > 0:
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_a_norm = nn.RMSNorm(q_lora_rank)
            self.q_b_proj = nn.Linear(
                q_lora_rank,
                num_heads * (qk_nope_head_dim + qk_rope_head_dim),
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                hidden_size,
                num_heads * (qk_nope_head_dim + qk_rope_head_dim),
                bias=False,
            )

        # KV compression
        self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_norm = nn.RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_kv_heads * (qk_nope_head_dim + head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # RoPE for the rotary-embedded portion
        self.rope = initialize_rope(
            qk_rope_head_dim,
            rope_theta,
            False,
            rope_scaling,
            max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        # Query
        if self.q_lora_rank > 0:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)

        q = q.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        q_nope = q[:, :, :, :self.qk_nope_head_dim]
        q_rope = q[:, :, :, self.qk_nope_head_dim:]

        # Compressed KV
        kv_compressed = self.kv_a_proj(x)
        kv_latent = kv_compressed[:, :, :self.kv_lora_rank]
        k_rope_input = kv_compressed[:, :, self.kv_lora_rank:]

        kv_latent = self.kv_a_norm(kv_latent)
        kv = self.kv_b_proj(kv_latent)
        kv = kv.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

        k_nope = kv[:, :, :, :self.qk_nope_head_dim]
        v = kv[:, :, :, self.qk_nope_head_dim:]

        # Rotary portion
        k_rope = k_rope_input.reshape(B, L, 1, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q_rope = self.rope(q_rope, offset=offset)
        k_rope = self.rope(k_rope, offset=offset)

        # Combine nope and rope parts
        # Expand k_rope to match num_kv_heads
        k_rope = mx.broadcast_to(k_rope, k_nope.shape[:2] + k_rope.shape[2:])

        keys = mx.concatenate([k_nope, k_rope], axis=-1)
        queries = mx.concatenate([q_nope, q_rope], axis=-1)

        # KV cache
        if cache is not None:
            keys, v = cache.update_and_fetch(keys, v)

        # Attention
        output = scaled_dot_product_attention(queries, keys, v, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)
