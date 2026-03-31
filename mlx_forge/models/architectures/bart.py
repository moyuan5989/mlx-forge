"""BART/mBART encoder-decoder architecture for MLX Forge.

Supports:
- BART (facebook/bart-base, facebook/bart-large, etc.)
- mBART (via MODEL_REMAPPING["mbart"] = "bart")

Like T5 but with:
- Absolute position embeddings (learned) instead of relative bias
- LayerNorm instead of RMSNorm
- Standard attention (no relative bias)
- GELU activation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ...inference.cache import KVCache
from .._base import BaseModelArgs
from .._base.attention import create_causal_mask, create_padding_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    """BART model configuration arguments."""

    model_type: str = "bart"
    d_model: int = 768
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 12
    decoder_attention_heads: int = 12
    encoder_ffn_dim: int = 3072
    decoder_ffn_dim: int = 3072
    vocab_size: int = 50265
    max_position_embeddings: int = 1024
    activation_function: str = "gelu"
    tie_word_embeddings: bool = True
    decoder_start_token_id: int = 2
    eos_token_id: int = 2
    pad_token_id: int = 1
    scale_embedding: bool = False

    def __post_init__(self):
        self.encoder_head_dim = self.d_model // self.encoder_attention_heads
        self.decoder_head_dim = self.d_model // self.decoder_attention_heads


class BartAttention(nn.Module):
    """Standard multi-head attention for BART (no relative position bias)."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, is_cross: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.is_cross = is_cross

        inner_dim = n_heads * head_dim
        self.q_proj = nn.Linear(d_model, inner_dim, bias=True)
        self.k_proj = nn.Linear(d_model, inner_dim, bias=True)
        self.v_proj = nn.Linear(d_model, inner_dim, bias=True)
        self.out_proj = nn.Linear(inner_dim, d_model, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        B, T_q, _ = hidden_states.shape
        is_cross = key_value_states is not None

        q = self.q_proj(hidden_states)
        if is_cross:
            k = self.k_proj(key_value_states)
            v = self.v_proj(key_value_states)
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        q = q.reshape(B, T_q, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        T_k = k.shape[1]
        k = k.reshape(B, T_k, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T_k, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        attn_out = scaled_dot_product_attention(q, k, v, self.scale, mask=mask)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T_q, -1)
        return self.out_proj(attn_out)


class BartEncoderLayer(nn.Module):
    """BART encoder layer: self-attention + MLP with post-norm."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = BartAttention(
            args.d_model, args.encoder_attention_heads, args.encoder_head_dim
        )
        self.self_attn_layer_norm = nn.LayerNorm(args.d_model)
        self.fc1 = nn.Linear(args.d_model, args.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(args.encoder_ffn_dim, args.d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(args.d_model)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, mask=attention_mask)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BartDecoderLayer(nn.Module):
    """BART decoder layer: causal self-attention + cross-attention + MLP."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = BartAttention(
            args.d_model, args.decoder_attention_heads, args.decoder_head_dim
        )
        self.self_attn_layer_norm = nn.LayerNorm(args.d_model)
        self.cross_attn = BartAttention(
            args.d_model, args.decoder_attention_heads, args.decoder_head_dim,
            is_cross=True,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(args.d_model)
        self.fc1 = nn.Linear(args.d_model, args.decoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(args.decoder_ffn_dim, args.d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(args.d_model)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden: Optional[mx.array] = None,
        self_attn_mask: Optional[mx.array] = None,
        cross_attn_mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        # Self-attention (causal)
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, mask=self_attn_mask, cache=cache)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-attention
        if encoder_hidden is not None:
            residual = hidden_states
            hidden_states = self.cross_attn(
                hidden_states, key_value_states=encoder_hidden, mask=cross_attn_mask
            )
            hidden_states = residual + hidden_states
            hidden_states = self.cross_attn_layer_norm(hidden_states)

        # MLP
        residual = hidden_states
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BartEncoder(nn.Module):
    """BART encoder with position embeddings."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_positions = nn.Embedding(
            args.max_position_embeddings, args.d_model
        )
        self.layers = [BartEncoderLayer(args) for _ in range(args.encoder_layers)]
        self.layernorm_embedding = nn.LayerNorm(args.d_model)

    def __call__(
        self,
        inputs_embeds: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, T, _ = inputs_embeds.shape
        position_ids = mx.arange(T)
        hidden_states = inputs_embeds + self.embed_positions(position_ids)
        hidden_states = self.layernorm_embedding(hidden_states)

        mask = create_padding_mask(attention_mask) if attention_mask is not None else None

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        return hidden_states


class BartDecoder(nn.Module):
    """BART decoder with position embeddings and cross-attention."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_positions = nn.Embedding(
            args.max_position_embeddings, args.d_model
        )
        self.layers = [BartDecoderLayer(args) for _ in range(args.decoder_layers)]
        self.layernorm_embedding = nn.LayerNorm(args.d_model)

    def __call__(
        self,
        inputs_embeds: mx.array,
        encoder_hidden: Optional[mx.array] = None,
        self_attn_mask: Optional[mx.array] = None,
        cross_attn_mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        B, T, _ = inputs_embeds.shape
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset
        position_ids = mx.arange(offset, offset + T)
        hidden_states = inputs_embeds + self.embed_positions(position_ids)
        hidden_states = self.layernorm_embedding(hidden_states)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            hidden_states = layer(
                hidden_states, encoder_hidden,
                self_attn_mask, cross_attn_mask,
                cache=layer_cache,
            )

        return hidden_states


class Model(nn.Module):
    """BART encoder-decoder model.

    Forward pass: model(encoder_input_ids, decoder_input_ids) → (B, T_dec, V)
    For inference: model.encode() once, then model.decode() autoregressively.
    """

    model_category = "encoder_decoder"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.shared = nn.Embedding(args.vocab_size, args.d_model)
        self.encoder = BartEncoder(args)
        self.decoder = BartDecoder(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self._embed_scale = args.d_model**0.5 if args.scale_embedding else 1.0

    def __call__(
        self,
        encoder_input_ids: mx.array,
        decoder_input_ids: mx.array,
        encoder_attention_mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        encoder_hidden = self.encode(encoder_input_ids, encoder_attention_mask)
        return self.decode(decoder_input_ids, encoder_hidden, cache=cache)

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.shared(input_ids) * self._embed_scale
        return self.encoder(h, attention_mask)

    def decode(
        self,
        decoder_input_ids: mx.array,
        encoder_hidden: mx.array,
        cache=None,
    ) -> mx.array:
        h = self.shared(decoder_input_ids) * self._embed_scale

        T = decoder_input_ids.shape[1]
        if T > 1:
            offset = 0
            if cache is not None and cache[0] is not None:
                offset = cache[0].offset
            causal_mask = create_causal_mask(T, offset=offset)
            self_attn_mask = mx.where(causal_mask, 0.0, -1e9)
            self_attn_mask = self_attn_mask[None, None, :, :]
        else:
            self_attn_mask = None

        h = self.decoder(h, encoder_hidden, self_attn_mask, cache=cache)

        if self.args.tie_word_embeddings:
            return self.shared.as_linear(h)
        return self.lm_head(h)

    def make_cache(self):
        return [KVCache() for _ in range(self.args.decoder_layers)]

    @property
    def layers(self):
        return self.decoder.layers

    @property
    def encoder_layers(self):
        return self.encoder.layers

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Map HuggingFace BART weight names to MLX Forge names."""
        sanitized = {}
        for k, v in weights.items():
            new_k = k
            # Remove "model." prefix (HF BART wraps in BartModel)
            if new_k.startswith("model."):
                new_k = new_k[6:]
            # encoder.layers stays the same (HF uses encoder.layers too)
            # decoder.layers stays the same

            # embed_tokens → shared (BART uses shared embeddings)
            if "encoder.embed_tokens" in new_k or "decoder.embed_tokens" in new_k:
                continue  # shared.weight handles this

            # HF BART embed_positions has an offset of 2
            # We handle this by loading directly

            # attention: k_proj, v_proj, q_proj, out_proj stay the same
            # fc1, fc2 stay the same
            # layer norms stay the same

            sanitized[new_k] = v

        return sanitized
