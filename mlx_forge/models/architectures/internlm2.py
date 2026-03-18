"""InternLM 2 architecture for MLX Forge.

Supports:
- InternLM 2 (7B, 20B)
- InternLM 2.5

Key features:
- Standard transformer with fused wqkv projection
- No bias in attention
- RMSNorm, SwiGLU

Based on mlx-lm implementation (MIT License).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .._base.activations import swiglu
from .._base.rope import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """InternLM 2 model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    bias: bool = False
    attn_implementation: str = "eager"

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class Attention(nn.Module):
    """Multi-head attention with fused WQK projection."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        # InternLM2 uses fused qkv projection
        qkv_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.wqkv = nn.Linear(dim, qkv_dim, bias=args.bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.wqkv(x)

        # Split into Q, K, V
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        queries = qkv[:, :, :q_size]
        keys = qkv[:, :, q_size : q_size + kv_size]
        values = qkv[:, :, q_size + kv_size :]

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    """InternLM 2 transformer layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MLP(args.hidden_size, args.intermediate_size, bias=args.bias)
        self.attention_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        return h + r


class InternLM2Model(nn.Module):
    """InternLM 2 transformer backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.tok_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """InternLM 2 model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = InternLM2Model(args)
        if not args.tie_word_embeddings:
            self.output = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.tok_embeddings.as_linear(out)
        else:
            out = self.output(out)
        return out

    def sanitize(self, weights: dict) -> dict:
        weights = {k: v for k, v in weights.items() if "rotary_emb" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("output.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers
