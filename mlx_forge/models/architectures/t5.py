"""T5/mT5 encoder-decoder architecture for MLX Forge.

Supports:
- T5 (t5-small, t5-base, t5-large, etc.)
- mT5 (via MODEL_REMAPPING["mt5"] = "t5")
- Flan-T5

Encoder-decoder model: encoder (bidirectional) + decoder (causal + cross-attention).
T5 uses RMSNorm and relative position bias (learned, bucketed logarithmically).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs
from .._base.attention import create_padding_mask, scaled_dot_product_attention
from ...inference.cache import KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    """T5 model configuration arguments."""

    model_type: str = "t5"
    d_model: int = 512
    d_ff: int = 2048
    d_kv: int = 64
    num_heads: int = 8
    num_layers: int = 6
    num_decoder_layers: int = 6
    vocab_size: int = 32128
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    is_gated_act: bool = False
    dense_act_fn: str = "relu"
    tie_word_embeddings: bool = True
    decoder_start_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 0

    # Map HF names
    hidden_size: int = 0

    def __post_init__(self):
        if self.hidden_size == 0:
            self.hidden_size = self.d_model
        else:
            self.d_model = self.hidden_size


class T5RMSNorm(nn.Module):
    """T5-style RMSNorm (no bias, no subtraction of mean)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = (x * x).mean(axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return self.weight * x


def _relative_position_bucket(
    relative_position: mx.array,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> mx.array:
    """T5 relative position bucketing (logarithmic)."""
    relative_buckets = mx.zeros_like(relative_position)
    n = -relative_position

    if bidirectional:
        num_buckets //= 2
        relative_buckets = relative_buckets + mx.where(n < 0, num_buckets, 0)
        n = mx.abs(n)
    else:
        n = mx.maximum(n, 0)

    max_exact = num_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
        mx.log(n.astype(mx.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(mx.int32)
    val_if_large = mx.minimum(val_if_large, num_buckets - 1)

    relative_buckets = relative_buckets + mx.where(is_small, n, val_if_large)
    return relative_buckets.astype(mx.int32)


class T5Attention(nn.Module):
    """T5 attention with relative position bias.

    Supports self-attention AND cross-attention (via key_value_states parameter).
    First layer computes bias; subsequent layers reuse it.
    """

    def __init__(self, args: ModelArgs, has_relative_bias: bool = False, is_cross: bool = False):
        super().__init__()
        self.n_heads = args.num_heads
        self.d_kv = args.d_kv
        self.scale = self.d_kv**-0.5
        self.is_cross = is_cross

        inner_dim = args.num_heads * args.d_kv
        self.q_proj = nn.Linear(args.d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(args.d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(args.d_model, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, args.d_model, bias=False)

        self.has_relative_bias = has_relative_bias
        if has_relative_bias:
            self.relative_attention_bias = nn.Embedding(
                args.relative_attention_num_buckets, args.num_heads
            )
            self.num_buckets = args.relative_attention_num_buckets
            self.max_distance = args.relative_attention_max_distance

    def _compute_bias(self, T_q: int, T_k: int, bidirectional: bool = True) -> mx.array:
        """Compute relative position bias of shape (1, n_heads, T_q, T_k)."""
        q_pos = mx.arange(T_q)[:, None]
        k_pos = mx.arange(T_k)[None, :]
        relative_position = k_pos - q_pos  # (T_q, T_k)

        buckets = _relative_position_bucket(
            relative_position,
            bidirectional=bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        # (T_q, T_k) → gather from embedding
        flat = buckets.reshape(-1)
        values = mx.take(self.relative_attention_bias.weight, flat, axis=0)
        values = values.reshape(T_q, T_k, self.n_heads)
        return values.transpose(2, 0, 1)[None, :, :, :]  # (1, H, T_q, T_k)

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
        cache=None,
    ):
        B, T_q, _ = hidden_states.shape
        is_cross = key_value_states is not None

        q = self.q_proj(hidden_states)
        if is_cross:
            k = self.k_proj(key_value_states)
            v = self.v_proj(key_value_states)
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        q = q.reshape(B, T_q, self.n_heads, self.d_kv).transpose(0, 2, 1, 3)
        T_k = k.shape[1]
        k = k.reshape(B, T_k, self.n_heads, self.d_kv).transpose(0, 2, 1, 3)
        v = v.reshape(B, T_k, self.n_heads, self.d_kv).transpose(0, 2, 1, 3)

        # KV cache for decoder self-attention
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
            T_k = k.shape[2]

        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Add position bias
        if self.has_relative_bias and position_bias is None:
            bidirectional = not (not is_cross and cache is not None)
            position_bias = self._compute_bias(T_q, T_k, bidirectional=bidirectional)

        if position_bias is not None:
            scores = scores + position_bias

        # Apply mask
        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        attn_out = attn_weights @ v

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T_q, -1)
        output = self.o_proj(attn_out)

        return output, position_bias


class T5MLP(nn.Module):
    """T5 feed-forward: supports gated (SwiGLU/GeGLU) and non-gated (ReLU)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.is_gated = args.is_gated_act
        self.wi = nn.Linear(args.d_model, args.d_ff, bias=False)
        if self.is_gated:
            self.wi_1 = nn.Linear(args.d_model, args.d_ff, bias=False)
        self.wo = nn.Linear(args.d_ff, args.d_model, bias=False)

        act_name = args.dense_act_fn
        if act_name == "gelu_new" or act_name == "gelu":
            self.act = nn.gelu
        elif act_name == "silu":
            self.act = nn.silu
        else:
            self.act = nn.relu

    def __call__(self, x: mx.array) -> mx.array:
        if self.is_gated:
            h = self.act(self.wi(x)) * self.wi_1(x)
        else:
            h = self.act(self.wi(x))
        return self.wo(h)


class T5EncoderLayer(nn.Module):
    """T5 encoder layer: self-attention + MLP with pre-norm."""

    def __init__(self, args: ModelArgs, has_relative_bias: bool = False):
        super().__init__()
        self.self_attn = T5Attention(args, has_relative_bias=has_relative_bias)
        self.norm1 = T5RMSNorm(args.d_model)
        self.mlp = T5MLP(args)
        self.norm2 = T5RMSNorm(args.d_model)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
    ):
        normed = self.norm1(hidden_states)
        attn_out, position_bias = self.self_attn(
            normed, mask=attention_mask, position_bias=position_bias
        )
        hidden_states = hidden_states + attn_out
        normed = self.norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(normed)
        return hidden_states, position_bias


class T5DecoderLayer(nn.Module):
    """T5 decoder layer: causal self-attention + cross-attention + MLP."""

    def __init__(self, args: ModelArgs, has_relative_bias: bool = False):
        super().__init__()
        self.self_attn = T5Attention(args, has_relative_bias=has_relative_bias)
        self.cross_attn = T5Attention(args, has_relative_bias=False, is_cross=True)
        self.norm1 = T5RMSNorm(args.d_model)
        self.norm2 = T5RMSNorm(args.d_model)
        self.mlp = T5MLP(args)
        self.norm3 = T5RMSNorm(args.d_model)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden: Optional[mx.array] = None,
        self_attn_mask: Optional[mx.array] = None,
        cross_attn_mask: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
        cache=None,
    ):
        # Self-attention (causal)
        normed = self.norm1(hidden_states)
        attn_out, position_bias = self.self_attn(
            normed, mask=self_attn_mask, position_bias=position_bias, cache=cache
        )
        hidden_states = hidden_states + attn_out

        # Cross-attention
        if encoder_hidden is not None:
            normed = self.norm2(hidden_states)
            cross_out, _ = self.cross_attn(
                normed, key_value_states=encoder_hidden, mask=cross_attn_mask
            )
            hidden_states = hidden_states + cross_out

        # MLP
        normed = self.norm3(hidden_states)
        hidden_states = hidden_states + self.mlp(normed)

        return hidden_states, position_bias


class T5Encoder(nn.Module):
    """T5 encoder: bidirectional self-attention stack."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = [
            T5EncoderLayer(args, has_relative_bias=(i == 0))
            for i in range(args.num_layers)
        ]
        self.final_norm = T5RMSNorm(args.d_model)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        position_bias = None
        for layer in self.layers:
            hidden_states, position_bias = layer(
                hidden_states, attention_mask, position_bias
            )
        return self.final_norm(hidden_states)


class T5Decoder(nn.Module):
    """T5 decoder: causal self-attention + cross-attention stack."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = [
            T5DecoderLayer(args, has_relative_bias=(i == 0))
            for i in range(args.num_decoder_layers)
        ]
        self.final_norm = T5RMSNorm(args.d_model)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden: Optional[mx.array] = None,
        self_attn_mask: Optional[mx.array] = None,
        cross_attn_mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        position_bias = None
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            hidden_states, position_bias = layer(
                hidden_states, encoder_hidden,
                self_attn_mask, cross_attn_mask,
                position_bias, cache=layer_cache,
            )
        return self.final_norm(hidden_states)


class Model(nn.Module):
    """T5 encoder-decoder model.

    Forward pass: model(encoder_input_ids, decoder_input_ids) → (B, T_dec, V)
    For inference: model.encode() once, then model.decode() autoregressively.
    """

    model_category = "encoder_decoder"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.shared = nn.Embedding(args.vocab_size, args.d_model)
        self.encoder = T5Encoder(args)
        self.decoder = T5Decoder(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def __call__(
        self,
        encoder_input_ids: mx.array,
        decoder_input_ids: mx.array,
        encoder_attention_mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        """Full forward pass for training.

        Args:
            encoder_input_ids: (B, T_enc) source token IDs
            decoder_input_ids: (B, T_dec) target token IDs
            encoder_attention_mask: (B, T_enc) padding mask
            cache: Optional KV cache for decoder

        Returns:
            (B, T_dec, V) logits
        """
        encoder_hidden = self.encode(encoder_input_ids, encoder_attention_mask)
        return self.decode(decoder_input_ids, encoder_hidden, cache=cache)

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode input sequence. Returns (B, T_enc, D) hidden states."""
        h = self.shared(input_ids)
        mask = create_padding_mask(attention_mask) if attention_mask is not None else None
        return self.encoder(h, mask)

    def decode(
        self,
        decoder_input_ids: mx.array,
        encoder_hidden: mx.array,
        cache=None,
    ) -> mx.array:
        """Decode with cross-attention to encoder hidden states.

        Returns: (B, T_dec, V) logits.
        """
        h = self.shared(decoder_input_ids)

        # Causal mask for decoder self-attention
        T = decoder_input_ids.shape[1]
        if T > 1:
            from .._base.attention import create_causal_mask
            offset = 0
            if cache is not None and cache[0] is not None:
                offset = cache[0].offset
            causal_mask = create_causal_mask(T, offset=offset)
            # Convert to additive mask: False → -inf
            self_attn_mask = mx.where(causal_mask, 0.0, -1e9)
            self_attn_mask = self_attn_mask[None, None, :, :]  # (1, 1, T, T+offset)
        else:
            self_attn_mask = None

        h = self.decoder(h, encoder_hidden, self_attn_mask, cache=cache)

        if self.args.tie_word_embeddings:
            return self.shared.as_linear(h)
        return self.lm_head(h)

    def make_cache(self):
        """Create KV cache for decoder layers only."""
        return [KVCache() for _ in range(self.args.num_decoder_layers)]

    @property
    def layers(self):
        """Return decoder layers for LoRA targeting."""
        return self.decoder.layers

    @property
    def encoder_layers(self):
        """Return encoder layers for LoRA targeting."""
        return self.encoder.layers

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Map HuggingFace T5 weight names to MLX Forge names."""
        sanitized = {}
        for k, v in weights.items():
            new_k = k

            # shared.weight stays as shared.weight
            # encoder.block.N → encoder.layers.N
            new_k = new_k.replace("encoder.block.", "encoder.layers.")
            new_k = new_k.replace("decoder.block.", "decoder.layers.")

            # layer.0.SelfAttention → self_attn
            new_k = new_k.replace(".layer.0.SelfAttention.", ".self_attn.")
            # layer.1.EncDecAttention → cross_attn
            new_k = new_k.replace(".layer.1.EncDecAttention.", ".cross_attn.")
            # layer.1.DenseReluDense or layer.2.DenseReluDense → mlp
            new_k = new_k.replace(".layer.1.DenseReluDense.", ".mlp.")
            new_k = new_k.replace(".layer.2.DenseReluDense.", ".mlp.")

            # LayerNorms
            new_k = new_k.replace(".layer.0.layer_norm.", ".norm1.")
            new_k = new_k.replace(".layer.1.layer_norm.", ".norm2.")
            new_k = new_k.replace(".layer.2.layer_norm.", ".norm3.")

            # T5 attention projections: q, k, v, o → q_proj, k_proj, v_proj, o_proj
            new_k = new_k.replace(".self_attn.q.", ".self_attn.q_proj.")
            new_k = new_k.replace(".self_attn.k.", ".self_attn.k_proj.")
            new_k = new_k.replace(".self_attn.v.", ".self_attn.v_proj.")
            new_k = new_k.replace(".self_attn.o.", ".self_attn.o_proj.")
            new_k = new_k.replace(".cross_attn.q.", ".cross_attn.q_proj.")
            new_k = new_k.replace(".cross_attn.k.", ".cross_attn.k_proj.")
            new_k = new_k.replace(".cross_attn.v.", ".cross_attn.v_proj.")
            new_k = new_k.replace(".cross_attn.o.", ".cross_attn.o_proj.")

            # Relative attention bias
            new_k = new_k.replace(
                ".self_attn.relative_attention_bias.",
                ".self_attn.relative_attention_bias."
            )

            # Final layer norm
            new_k = new_k.replace("encoder.final_layer_norm.", "encoder.final_norm.")
            new_k = new_k.replace("decoder.final_layer_norm.", "decoder.final_norm.")

            # MLP weight names: wi → wi, wi_0 → wi, wi_1 → wi_1, wo → wo
            new_k = new_k.replace(".mlp.wi_0.", ".mlp.wi.")

            # lm_head stays
            # Drop decoder.embed_tokens (uses shared)
            if "decoder.embed_tokens" in new_k or "encoder.embed_tokens" in new_k:
                continue

            sanitized[new_k] = v

        return sanitized
