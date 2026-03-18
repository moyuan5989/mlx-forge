"""ChatGLM-4 architecture for MLX Forge.

Supports:
- ChatGLM-4, GLM-4

Key features:
- RMSNorm, SwiGLU activation
- add_bias_linear, add_qkv_bias parameters
- Multi-query attention support

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
    """GLM-4 model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float = 1e-5
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 131072
    rope_theta: float = 10000
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    padded_vocab_size: Optional[int] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.padded_vocab_size is not None:
            self.vocab_size = self.padded_vocab_size


class Attention(nn.Module):
    """Multi-head attention with optional QKV bias."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        qkv_bias = args.add_qkv_bias
        linear_bias = args.add_bias_linear

        # Fused QKV projection
        qkv_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(dim, qkv_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=linear_bias)

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

        qkv = self.qkv_proj(x)

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
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP for GLM-4."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    """GLM-4 transformer layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size, bias=args.add_bias_linear)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
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


class GLM4Model(nn.Module):
    """GLM-4 transformer backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """GLM-4 model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GLM4Model(args)
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
        # Handle transformer prefix remapping
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("transformer."):
                k = k.replace("transformer.", "model.", 1)
                k = k.replace("embedding.word_embeddings.", "embed_tokens.", 1)
                k = k.replace("encoder.layers.", "layers.", 1)
                k = k.replace("encoder.final_layernorm.", "norm.", 1)
                k = k.replace("output_layer.", "lm_head.", 1)
            new_weights[k] = v
        return new_weights

    @property
    def layers(self):
        return self.model.layers
