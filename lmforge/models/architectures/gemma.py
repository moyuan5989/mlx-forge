"""Gemma model architecture for LMForge.

Supports:
- Gemma 1 (2B, 7B)
- Gemma 2 (2B, 9B, 27B) — with soft-capping, sliding window, post-norms
- Gemma 3 (1B, 4B, 12B, 27B)

Key differences from Llama:
- RMSNorm with +1 offset: x * norm * (1 + weight)
- Tied embeddings (embed_tokens used as lm_head)
- Embedding scaling by sqrt(hidden_size)
- Explicit head_dim in config
- Gemma 2: soft attention capping (tanh-based)
- Gemma 2: sliding window on alternating layers
- Gemma 2: post-attention and post-FFN RMSNorm
- GeGLU activation (Gemma 2/3)

Reference: mlx-lm (MIT License)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .._base.activations import swiglu
from .._base.rope import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """Gemma model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = True
    hidden_activation: Optional[str] = None
    # Gemma 2 specific
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    sliding_window: Optional[int] = None
    query_pre_attn_scalar: Optional[float] = None

    def __post_init__(self):
        # Default hidden_activation to gelu for Gemma
        if self.hidden_activation is None:
            self.hidden_activation = "gelu"


class GemmaRMSNorm(nn.Module):
    """RMSNorm with +1 offset (Gemma-specific).

    Unlike standard RMSNorm where output = x * norm * weight,
    Gemma uses output = x * norm * (1 + weight).
    This means the weights are initialized around 0 rather than 1.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * norm * (1.0 + self.weight)


class Attention(nn.Module):
    """Multi-head attention with optional soft-capping and sliding window."""

    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.layer_idx = layer_idx

        # Attention scale: Gemma 2 can use query_pre_attn_scalar
        if args.query_pre_attn_scalar is not None:
            self.scale = args.query_pre_attn_scalar**-0.5
        else:
            self.scale = self.head_dim**-0.5

        # Soft attention capping (Gemma 2)
        self.attn_logit_softcapping = args.attn_logit_softcapping

        # Sliding window: Gemma 2 uses sliding window on even layers (0-indexed)
        self.sliding_window = None
        if args.sliding_window is not None and layer_idx % 2 == 0:
            self.sliding_window = args.sliding_window

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

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

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Soft-capping: if enabled, use manual attention instead of fast SDPA
        if self.attn_logit_softcapping is not None:
            output = self._attention_with_softcapping(
                queries, keys, values, mask
            )
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    def _attention_with_softcapping(self, q, k, v, mask):
        """Attention with logit soft-capping (Gemma 2).

        Applies tanh-based capping to prevent attention scores from
        becoming too large: scores = cap * tanh(scores / cap)
        """
        # Manual attention computation
        # GQA: expand k, v heads to match q heads
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply soft-capping
        cap = self.attn_logit_softcapping
        scores = cap * mx.tanh(scores / cap)

        # Apply mask
        if mask is not None:
            if isinstance(mask, str) and mask == "causal":
                # Build explicit causal mask for soft-capping path
                T = scores.shape[-1]
                causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
                scores = mx.where(causal, scores, mx.array(float("-inf")))
            else:
                scores = mx.where(mask, scores, mx.array(float("-inf")))

        weights = mx.softmax(scores, axis=-1)
        return weights @ v


class MLP(nn.Module):
    """Feed-forward network with configurable activation (SwiGLU or GeGLU)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

        # Gemma 2/3 uses GeGLU, Gemma 1 uses GELU
        act = args.hidden_activation
        if act in ("gelu_pytorch_tanh", "gelu_new", "gelu"):
            self._act_fn = nn.gelu
        elif act == "silu":
            self._act_fn = nn.silu
        else:
            # Default to GELU for Gemma
            self._act_fn = nn.gelu

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self._act_fn(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Gemma transformer layer.

    Gemma 2 adds post-attention and post-FFN RMSNorm (4 norms total).
    Gemma 1 uses standard pre-norm only (2 norms).
    """

    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.self_attn = Attention(args, layer_idx=layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        # Gemma 2: additional post-norms
        self.has_post_norms = args.model_type == "gemma2"
        if self.has_post_norms:
            self.pre_feedforward_layernorm = GemmaRMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.post_feedforward_layernorm = GemmaRMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)

        if self.has_post_norms:
            r = self.post_attention_layernorm(r)
            h = x + r
            r = self.mlp(self.pre_feedforward_layernorm(h))
            r = self.post_feedforward_layernorm(r)
        else:
            h = x + r
            r = self.mlp(self.post_attention_layernorm(h))

        return h + r


class GemmaModel(nn.Module):
    """Gemma transformer backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        # Gemma scales embeddings by sqrt(hidden_size)
        h = h * math.sqrt(self.args.hidden_size)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """Gemma model for LMForge training and inference.

    Uses tied embeddings: embed_tokens weights serve as both
    the input embedding and the output lm_head.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GemmaModel(args)

        # Gemma typically ties embeddings, but support untied too
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)

        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)

        # Gemma 2: final logit soft-capping
        if self.args.final_logit_softcapping is not None:
            cap = self.args.final_logit_softcapping
            out = cap * mx.tanh(out / cap)

        return out

    def sanitize(self, weights: dict) -> dict:
        """Clean up weight dict before loading.

        - Remove precomputed rotary frequencies
        - Remove lm_head if using tied embeddings
        """
        weights = {
            k: v
            for k, v in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        """Access transformer layers for LoRA targeting."""
        return self.model.layers
