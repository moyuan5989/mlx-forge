"""Llama 4 architecture for MLX Forge.

Supports:
- Llama 4 Scout, Maverick

Key features:
- Native MoE support with top-k routing
- Chunked attention (interleaved local/global attention)
- Temperature-scaled softmax for router

Based on mlx-lm implementation (MIT License).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .._base.activations import swiglu
from .._base.attention import create_causal_mask
from .._base.rope import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """Llama 4 model configuration arguments."""

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
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    # MoE parameters
    num_local_experts: int = 1
    num_experts_per_tok: int = 1
    router_temperature: float = 1.0

    # Chunked attention
    attn_chunk_size: Optional[int] = None
    no_rope_layers: Optional[list] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class Attention(nn.Module):
    """Multi-head attention with optional chunked attention."""

    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.attn_chunk_size = args.attn_chunk_size

        # Some layers may not use RoPE
        self.use_rope = True
        if args.no_rope_layers is not None and layer_idx in args.no_rope_layers:
            self.use_rope = False

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.attention_bias)

        if self.use_rope:
            self.rope = initialize_rope(
                self.head_dim,
                args.rope_theta,
                args.rope_traditional,
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

        if self.use_rope:
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Apply chunked attention mask if needed
        if self.attn_chunk_size is not None and L > 1:
            offset = cache.offset - L if cache is not None else 0
            chunk_mask = create_causal_mask(
                L, offset=max(0, offset), window_size=self.attn_chunk_size
            )
            if mask is not None and not isinstance(mask, str):
                chunk_mask = mask & chunk_mask
            mask = chunk_mask

        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Expert(nn.Module):
    """Single expert MLP."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoELayer(nn.Module):
    """Llama 4 MoE layer with temperature-scaled routing."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.router_temperature = args.router_temperature

        self.gate = nn.Linear(args.hidden_size, self.num_experts, bias=False)
        self.experts = [
            Expert(args.hidden_size, args.intermediate_size)
            for _ in range(self.num_experts)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)

        router_logits = self.gate(x_flat)

        # Temperature-scaled softmax
        if self.router_temperature != 1.0:
            router_logits = router_logits / self.router_temperature

        top_k_indices = mx.argpartition(
            -router_logits, kth=self.num_experts_per_tok - 1, axis=-1
        )[:, : self.num_experts_per_tok]
        top_k_logits = mx.take_along_axis(router_logits, top_k_indices, axis=-1)
        top_k_weights = mx.softmax(top_k_logits, axis=-1)

        output = mx.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            expert_mask = top_k_indices == i
            token_mask = mx.any(expert_mask, axis=-1)
            if not mx.any(token_mask).item():
                continue
            expert_weights = mx.sum(
                mx.where(expert_mask, top_k_weights, mx.zeros_like(top_k_weights)),
                axis=-1,
                keepdims=True,
            )
            expert_out = expert(x_flat)
            output = output + expert_out * expert_weights

        return output.reshape(B, L, D)


class TransformerBlock(nn.Module):
    """Llama 4 transformer layer (dense or MoE)."""

    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.self_attn = Attention(args, layer_idx=layer_idx)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_local_experts > 1:
            self.mlp = MoELayer(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)

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


class Llama4Model(nn.Module):
    """Llama 4 transformer backbone."""

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
    """Llama 4 model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Llama4Model(args)
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
