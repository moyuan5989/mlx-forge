"""DeepSeek V2 architecture for MLX Forge.

Supports:
- DeepSeek-V2, DeepSeek-V2-Lite

Key features:
- Multi-Latent Attention (MLA): compressed KV via low-rank projection
- MoE layers with shared experts
- Some layers are dense, others are MoE

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
    """DeepSeek V2 model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool = False
    rope_theta: float = 10000
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 4096

    # MLA parameters
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    num_key_value_heads: Optional[int] = None

    # MoE parameters
    num_experts: int = 1
    num_experts_per_tok: int = 1
    num_shared_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    first_k_dense_replace: int = 0
    norm_topk_prob: bool = True
    moe_layer_freq: int = 1

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim


class MLAAttention(nn.Module):
    """Multi-Latent Attention with compressed KV projection."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scale = self.qk_head_dim**-0.5

        dim = args.hidden_size

        # Query projection
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(dim, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                dim, self.n_heads * self.qk_head_dim, bias=False
            )

        # KV compressed projection
        self.kv_a_proj_with_mqa = nn.Linear(
            dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.n_heads * self.v_head_dim, dim, bias=False
        )

        self.rope = initialize_rope(
            self.qk_rope_head_dim,
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

        # Query
        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)

        q = q.reshape(B, L, self.n_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_rope = mx.split(
            q, [self.qk_nope_head_dim], axis=-1
        )

        # Compressed KV
        kv_combined = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_rope = mx.split(
            kv_combined, [self.kv_lora_rank], axis=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)

        # Decompress KV
        kv = self.kv_b_proj(compressed_kv)
        kv = kv.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(
            kv, [self.qk_nope_head_dim], axis=-1
        )

        # Reshape k_rope for RoPE: (B, L, 1, rope_dim) -> (B, 1, L, rope_dim)
        k_rope = k_rope.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            q_rope = self.rope(q_rope, offset=cache.offset)
            k_rope = self.rope(k_rope, offset=cache.offset)
        else:
            q_rope = self.rope(q_rope)
            k_rope = self.rope(k_rope)

        # Expand k_rope to all heads
        k_rope = mx.repeat(k_rope, self.n_heads, axis=1)

        # Combine nope and rope parts
        queries = mx.concatenate([q_nope, q_rope], axis=-1)
        keys = mx.concatenate([k_nope, k_rope], axis=-1)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """Standard SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Expert(nn.Module):
    """Single expert MLP for MoE."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoELayer(nn.Module):
    """DeepSeek V2 MoE layer with shared experts."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(args.hidden_size, self.num_experts, bias=False)
        self.experts = [
            Expert(args.hidden_size, args.moe_intermediate_size)
            for _ in range(self.num_experts)
        ]

        if args.num_shared_experts is not None and args.num_shared_experts > 0:
            shared_intermediate = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = MLP(args.hidden_size, shared_intermediate)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)

        router_logits = self.gate(x_flat)
        top_k_indices = mx.argpartition(
            -router_logits, kth=self.num_experts_per_tok - 1, axis=-1
        )[:, : self.num_experts_per_tok]
        top_k_logits = mx.take_along_axis(router_logits, top_k_indices, axis=-1)
        top_k_weights = mx.softmax(top_k_logits, axis=-1)

        if not self.norm_topk_prob:
            top_k_weights = top_k_weights * mx.softmax(router_logits, axis=-1).max(
                axis=-1, keepdims=True
            )

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

        output = output.reshape(B, L, D)

        if self.shared_experts is not None:
            output = output + self.shared_experts(x)

        return output


class TransformerBlock(nn.Module):
    """DeepSeek V2 transformer layer (dense or MoE)."""

    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.self_attn = MLAAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        # Use MoE for layers after first_k_dense_replace, at moe_layer_freq intervals
        use_moe = (
            layer_idx >= args.first_k_dense_replace
            and args.num_experts > 1
            and (layer_idx - args.first_k_dense_replace) % args.moe_layer_freq == 0
        )
        if use_moe:
            self.mlp = MoELayer(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

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


class DeepSeekV2Model(nn.Module):
    """DeepSeek V2 transformer backbone."""

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
    """DeepSeek V2 model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepSeekV2Model(args)
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
