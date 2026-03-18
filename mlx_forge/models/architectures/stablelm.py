"""StableLM architecture for MLX Forge.

Supports:
- StableLM-2, StableLM Zephyr
- StableLM 3B, 1.6B

Key features:
- Partial RoPE (rotary_pct parameter)
- Uses both RoPE and non-rotary dimensions
- LayerNorm (not RMSNorm)

Based on mlx-lm implementation (MIT License).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .._base.rope import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """StableLM model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 4096
    rope_theta: float = 10000
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    use_qkv_bias: bool = False
    layer_norm_eps: float = 1e-5
    rotary_pct: float = 0.25
    hidden_act: str = "silu"
    use_cache: bool = True
    norm_eps: Optional[float] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.norm_eps is not None:
            self.layer_norm_eps = self.norm_eps


class Attention(nn.Module):
    """Multi-head attention with partial RoPE."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        # Partial rotary: only apply RoPE to a fraction of head_dim
        self.rotary_dim = int(self.head_dim * args.rotary_pct)

        qkv_bias = args.use_qkv_bias

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.rope = initialize_rope(
            self.rotary_dim,
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

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply partial RoPE: split into rotary and pass-through dimensions
        q_rot, q_pass = queries[..., : self.rotary_dim], queries[..., self.rotary_dim :]
        k_rot, k_pass = keys[..., : self.rotary_dim], keys[..., self.rotary_dim :]

        if cache is not None:
            q_rot = self.rope(q_rot, offset=cache.offset)
            k_rot = self.rope(k_rot, offset=cache.offset)
        else:
            q_rot = self.rope(q_rot)
            k_rot = self.rope(k_rot)

        queries = mx.concatenate([q_rot, q_pass], axis=-1)
        keys = mx.concatenate([k_rot, k_pass], axis=-1)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """Feed-forward network with configurable activation."""

    def __init__(self, dim: int, hidden_dim: int, act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

        if act == "silu":
            self._act_fn = nn.silu
        elif act == "gelu":
            self._act_fn = nn.gelu
        else:
            self._act_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self._act_fn(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """StableLM transformer layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size, act=args.hidden_act)
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class StableLMModel(nn.Module):
    """StableLM transformer backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """StableLM model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = StableLMModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights: dict) -> dict:
        weights = {k: v for k, v in weights.items() if "rotary_emb" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers
