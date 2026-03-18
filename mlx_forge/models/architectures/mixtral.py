"""Mixtral MoE architecture for MLX Forge.

Supports:
- Mixtral 8x7B, 8x22B
- Sparse Mixture of Experts with top-k routing

Key features:
- SwitchGLU MoE: top-k expert routing with softmax gating
- Each expert is a standard SwiGLU MLP
- Shared attention layers with standard GQA

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
    """Mixtral model configuration arguments."""

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
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    router_aux_loss_coef: float = 0.001

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class Attention(nn.Module):
    """Multi-head attention with GQA support."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            head_dim,
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


class Expert(nn.Module):
    """Single SwiGLU expert MLP."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok

        self.gate = nn.Linear(args.hidden_size, self.num_experts, bias=False)
        self.experts = [
            Expert(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
            for _ in range(self.num_experts)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape

        # Flatten batch and sequence
        x_flat = x.reshape(-1, D)  # (B*L, D)

        # Router: compute gating scores
        router_logits = self.gate(x_flat)  # (B*L, num_experts)

        # Top-k expert selection
        top_k_indices = mx.argpartition(
            -router_logits, kth=self.num_experts_per_tok - 1, axis=-1
        )[:, : self.num_experts_per_tok]  # (B*L, k)

        # Gather top-k logits and compute softmax weights
        top_k_logits = mx.take_along_axis(
            router_logits, top_k_indices, axis=-1
        )
        top_k_weights = mx.softmax(top_k_logits, axis=-1)  # (B*L, k)

        # Dispatch to experts and combine
        output = mx.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Build mask for tokens routed to this expert
            expert_mask = top_k_indices == i  # (B*L, k)
            # Check if any token is routed to this expert
            token_mask = mx.any(expert_mask, axis=-1)  # (B*L,)

            if not mx.any(token_mask).item():
                continue

            # Get weights for this expert
            expert_weights = mx.sum(
                mx.where(expert_mask, top_k_weights, mx.zeros_like(top_k_weights)),
                axis=-1,
                keepdims=True,
            )  # (B*L, 1)

            # Compute expert output and weight it
            expert_out = expert(x_flat)  # (B*L, D)
            output = output + expert_out * expert_weights

        return output.reshape(B, L, D)


class TransformerBlock(nn.Module):
    """Mixtral transformer layer with MoE MLP."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.block_sparse_moe = MoELayer(args)
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
        r = self.block_sparse_moe(self.post_attention_layernorm(h))
        return h + r


class MixtralModel(nn.Module):
    """Mixtral transformer backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
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
    """Mixtral MoE model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MixtralModel(args)
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
