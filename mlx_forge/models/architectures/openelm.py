"""Apple OpenELM architecture for MLX Forge.

Supports:
- OpenELM 270M, 450M, 1.1B, 3B

Key features:
- Layer-wise scaling of head dims and FFN dims
- num_query_heads and num_kv_heads can vary per layer
- ffn_dim_divisor for rounding FFN dimensions
- No bias anywhere

Based on mlx-lm implementation (MIT License).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .._base.rope import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """OpenELM model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 2048
    rope_theta: float = 10000
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    head_dim: int = 64

    # Per-layer head counts
    num_query_heads: Optional[List[int]] = None
    num_kv_heads: Optional[List[int]] = None

    # Per-layer FFN dims
    ffn_intermediate_size: Optional[List[int]] = None
    ffn_dim_divisor: int = 256

    # Fallback uniform values
    num_attention_heads: int = 12
    intermediate_size: int = 0
    ffn_multipliers: Optional[List[float]] = None

    def __post_init__(self):
        # Build per-layer head counts if not provided
        if self.num_query_heads is None:
            self.num_query_heads = [self.num_attention_heads] * self.num_hidden_layers
        if self.num_kv_heads is None:
            self.num_kv_heads = [1] * self.num_hidden_layers

        # Build per-layer FFN dims if not provided
        if self.ffn_intermediate_size is None:
            if self.ffn_multipliers is not None:
                self.ffn_intermediate_size = []
                for mult in self.ffn_multipliers:
                    ffn_dim = int(self.hidden_size * mult)
                    # Round to ffn_dim_divisor
                    if self.ffn_dim_divisor > 0:
                        ffn_dim = (
                            (ffn_dim + self.ffn_dim_divisor - 1)
                            // self.ffn_dim_divisor
                            * self.ffn_dim_divisor
                        )
                    self.ffn_intermediate_size.append(ffn_dim)
            else:
                if self.intermediate_size > 0:
                    self.ffn_intermediate_size = [self.intermediate_size] * self.num_hidden_layers
                else:
                    self.ffn_intermediate_size = [self.hidden_size * 4] * self.num_hidden_layers


class Attention(nn.Module):
    """Multi-head attention with per-layer head configuration."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = args.num_query_heads[layer_idx]
        self.n_kv_heads = args.num_kv_heads[layer_idx]
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # QK normalization
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

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

        # QK normalization
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

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
    """SwiGLU MLP with per-layer dimensions."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """OpenELM transformer layer with per-layer configuration."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx=layer_idx)
        self.mlp = MLP(args.hidden_size, args.ffn_intermediate_size[layer_idx])
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


class OpenELMModel(nn.Module):
    """OpenELM transformer backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args, layer_idx=i) for i in range(args.num_hidden_layers)
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
    """Apple OpenELM model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OpenELMModel(args)
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
        # Remap OpenELM-specific weight names
        new_weights = {}
        for k, v in weights.items():
            # Map "transformer." prefix to "model."
            if k.startswith("transformer."):
                k = k.replace("transformer.", "model.", 1)
                k = k.replace("token_embeddings.", "embed_tokens.", 1)
            new_weights[k] = v
        if self.args.tie_word_embeddings:
            new_weights.pop("lm_head.weight", None)
        return new_weights

    @property
    def layers(self):
        return self.model.layers
