"""DeBERTa v2/v3 encoder architecture for MLX Forge.

Supports:
- DeBERTa v2 (microsoft/deberta-v2-*)
- DeBERTa v3 (microsoft/deberta-v3-*)

Key difference from BERT: disentangled attention separating content and position.
Uses relative position bias instead of absolute position embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs
from .._base.attention import create_padding_mask


@dataclass
class ModelArgs(BaseModelArgs):
    """DeBERTa model configuration arguments."""

    model_type: str = "deberta"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 128100
    type_vocab_size: int = 0
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-7
    hidden_act: str = "gelu"
    relative_attention: bool = True
    max_relative_positions: int = 256
    position_buckets: int = 256
    pos_att_type: str = "p2c|c2p"

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads


class DeBERTaEmbeddings(nn.Module):
    """DeBERTa embeddings: token embeddings + LayerNorm (no position embeddings)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        if args.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                args.type_vocab_size, args.hidden_size
            )
        else:
            self.token_type_embeddings = None
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        embeddings = self.word_embeddings(input_ids)
        if self.token_type_embeddings is not None and token_type_ids is not None:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        return self.LayerNorm(embeddings)


class DisentangledAttention(nn.Module):
    """DeBERTa disentangled attention with content-to-position and position-to-content scores."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.max_relative_positions = args.max_relative_positions

        dim = args.hidden_size
        self.query = nn.Linear(dim, dim, bias=True)
        self.key = nn.Linear(dim, dim, bias=True)
        self.value = nn.Linear(dim, dim, bias=True)
        self.dense = nn.Linear(dim, dim, bias=True)
        self.LayerNorm = nn.LayerNorm(dim, eps=args.layer_norm_eps)

        # Relative position embeddings for disentangled attention
        self.rel_embeddings = nn.Embedding(
            2 * args.max_relative_positions, args.hidden_size
        )

        self.pos_att_type = args.pos_att_type.split("|")

    def _get_relative_positions(self, T: int) -> mx.array:
        """Compute relative position indices clipped to [-max_rel, max_rel].

        Returns: (T, T) array of indices into rel_embeddings (shifted to [0, 2*max_rel)).
        """
        positions = mx.arange(T)
        # (T, 1) - (1, T) = (T, T) relative distances
        rel_pos = positions[:, None] - positions[None, :]
        # Clip and shift to positive indices
        max_rel = self.max_relative_positions
        rel_pos = mx.clip(rel_pos, -max_rel + 1, max_rel - 1)
        rel_pos = rel_pos + (max_rel - 1)
        return rel_pos

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, T, _ = hidden_states.shape

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Content-to-content attention
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Relative position bias
        rel_pos = self._get_relative_positions(T)  # (T, T)
        # Use mx.take for indexing (MLX fancy indexing workaround)
        rel_emb = self.rel_embeddings.weight  # (2*max_rel, D)
        # Flatten rel_pos to 1D, gather, reshape
        flat_pos = rel_pos.reshape(-1)
        pos_emb = mx.take(rel_emb, flat_pos, axis=0)  # (T*T, D)
        pos_emb = pos_emb.reshape(T, T, self.n_heads, self.head_dim)

        # Content-to-position (c2p)
        if "c2p" in self.pos_att_type:
            # q @ pos^T: (B, H, T, 1, d) @ (1, 1, T, d, T) isn't efficient
            # Instead: for each query position i, dot with pos_emb[i, :] over j
            # pos_emb is (T, T, H, d) — transpose to (H, T, T, d)
            pos_emb_t = pos_emb.transpose(2, 0, 1, 3)  # (H, T, T, d)
            # q is (B, H, T, d), we want q[b,h,i,:] . pos_emb[h,i,j,:] for all i,j
            # = einsum("bhid,hitd->bhit" where t=j) — use broadcasting
            c2p_scores = (q[:, :, :, None, :] * pos_emb_t[None, :, :, :, :]).sum(-1)
            attn_scores = attn_scores + c2p_scores * self.scale

        # Position-to-content (p2c)
        if "p2c" in self.pos_att_type:
            pos_emb_t = pos_emb.transpose(2, 0, 1, 3)  # (H, T, T, d)
            p2c_scores = (k[:, :, None, :, :] * pos_emb_t[None, :, :, :, :]).sum(-1)
            # p2c_scores is (B, H, T_q, T_k) — transpose last two dims
            p2c_scores = p2c_scores.transpose(0, 1, 3, 2)
            attn_scores = attn_scores + p2c_scores * self.scale

        # Apply padding mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax and apply to values
        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_out = attn_weights @ v

        # Reshape back to (B, T, D)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)

        # Output projection + residual + LayerNorm
        output = self.dense(attn_out)
        return self.LayerNorm(output + hidden_states)


class DeBERTaMLP(nn.Module):
    """DeBERTa feed-forward network."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size, bias=True)
        self.dense_out = nn.Linear(args.intermediate_size, args.hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        h = nn.gelu(self.dense(hidden_states))
        h = self.dense_out(h)
        return self.LayerNorm(h + hidden_states)


class DeBERTaLayer(nn.Module):
    """Single DeBERTa transformer layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = DisentangledAttention(args)
        self.mlp = DeBERTaMLP(args)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.attention(hidden_states, attention_mask)
        return self.mlp(h)


class DeBERTaEncoder(nn.Module):
    """Stack of DeBERTa transformer layers."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = [DeBERTaLayer(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class Model(nn.Module):
    """DeBERTa encoder model.

    Returns hidden states (B, T, D), not logits.
    Uses disentangled attention with relative position bias.
    """

    model_category = "encoder"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = DeBERTaEmbeddings(args)
        self.encoder = DeBERTaEncoder(args)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embeddings(input_ids, token_type_ids)
        mask = create_padding_mask(attention_mask) if attention_mask is not None else None
        return self.encoder(h, mask)

    @property
    def layers(self):
        return self.encoder.layers

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Map HuggingFace DeBERTa weight names to MLX Forge names."""
        sanitized = {}
        for k, v in weights.items():
            if "pooler" in k:
                continue
            if "position_ids" in k:
                continue

            new_k = k
            # Remove "deberta." prefix
            if new_k.startswith("deberta."):
                new_k = new_k[8:]
            # encoder.layer.N → encoder.layers.N
            new_k = new_k.replace("encoder.layer.", "encoder.layers.")
            # attention.self.X → attention.X
            new_k = new_k.replace("attention.self.", "attention.")
            # attention.output.dense → attention.dense
            new_k = new_k.replace("attention.output.dense", "attention.dense")
            new_k = new_k.replace("attention.output.LayerNorm", "attention.LayerNorm")
            # intermediate.dense → mlp.dense
            new_k = new_k.replace("intermediate.dense", "mlp.dense")
            # output.dense → mlp.dense_out
            new_k = new_k.replace("output.dense", "mlp.dense_out")
            new_k = new_k.replace("output.LayerNorm", "mlp.LayerNorm")
            # encoder.rel_embeddings → encoder.layers shared? No — DeBERTa has
            # per-layer or shared rel_embeddings. Map to attention level.
            # HF: deberta.encoder.rel_embeddings.weight
            # We store on each attention layer via sanitize replication
            # Actually, HF DeBERTa shares rel_embeddings across layers
            # We keep it at encoder level and each attention reads from it
            # But our model has per-attention rel_embeddings... let's handle this:
            # For simplicity, if it's encoder.rel_embeddings, we replicate to each layer
            if "encoder.rel_embeddings" in new_k:
                # Replicate to all layers
                for i in range(100):  # upper bound
                    layer_k = f"encoder.layers.{i}.attention.rel_embeddings.weight"
                    sanitized[layer_k] = v
                continue

            sanitized[new_k] = v

        return sanitized
