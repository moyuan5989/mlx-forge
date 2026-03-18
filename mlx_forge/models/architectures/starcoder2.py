"""StarCoder 2 architecture for MLX Forge.

Supports:
- StarCoder2-3B, 7B, 15B

Key features:
- Code model with GQA
- Similar to Llama but with optional sliding window
- Uses grouped query attention

Based on mlx-lm implementation (MIT License).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .._base.attention import create_causal_mask
from .._base.rope import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """StarCoder 2 model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 16384
    rope_theta: float = 10000
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = True
    mlp_bias: bool = True
    use_bias: bool = True
    norm_epsilon: float = 1e-5
    residual_in_fp32: bool = False
    sliding_window: Optional[int] = None
    hidden_act: str = "gelu_pytorch_tanh"

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class Attention(nn.Module):
    """Multi-head attention with GQA and optional sliding window."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.sliding_window = args.sliding_window

        bias = args.attention_bias

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)

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

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

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

        if self.sliding_window is not None and L > 1:
            offset = cache.offset - L if cache is not None else 0
            sw_mask = create_causal_mask(
                L, offset=max(0, offset), window_size=self.sliding_window
            )
            if mask is not None and not isinstance(mask, str):
                sw_mask = mask & sw_mask
            mask = sw_mask

        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """Feed-forward network with configurable activation."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        bias = args.mlp_bias

        self.c_fc = nn.Linear(dim, hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=bias)

        act = args.hidden_act
        if act in ("gelu_pytorch_tanh", "gelu_new"):
            self._act_fn = nn.gelu_approx
        elif act == "gelu":
            self._act_fn = nn.gelu
        elif act == "silu":
            self._act_fn = nn.silu
        elif act == "relu":
            self._act_fn = nn.relu
        else:
            self._act_fn = nn.gelu

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(self._act_fn(self.c_fc(x)))


class TransformerBlock(nn.Module):
    """StarCoder 2 transformer layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.norm_epsilon
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


class StarCoder2Model(nn.Module):
    """StarCoder 2 transformer backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.norm_epsilon)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """StarCoder 2 model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = StarCoder2Model(args)
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
