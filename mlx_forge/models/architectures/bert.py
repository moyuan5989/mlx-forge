"""BERT/RoBERTa encoder architecture for MLX Forge.

Supports:
- BERT (bert-base-uncased, bert-large-uncased, etc.)
- RoBERTa (via MODEL_REMAPPING["roberta"] = "bert")

Encoder-only model: bidirectional attention, no KV cache, no causal mask.
Output is hidden states (B, T, D), not logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs
from .._base.attention import create_padding_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    """BERT model configuration arguments."""

    model_type: str = "bert"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 30522
    type_vocab_size: int = 2
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads


class BertEmbeddings(nn.Module):
    """BERT embeddings: token + position (absolute) + token_type → LayerNorm."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.position_embeddings = nn.Embedding(
            args.max_position_embeddings, args.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            args.type_vocab_size, args.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, T = input_ids.shape

        word_emb = self.word_embeddings(input_ids)

        position_ids = mx.arange(T)
        pos_emb = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)
        type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + pos_emb + type_emb
        return self.LayerNorm(embeddings)


class BertAttention(nn.Module):
    """Bidirectional multi-head attention (no causal mask, no RoPE, no KV cache)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        dim = args.hidden_size
        self.query = nn.Linear(dim, dim, bias=True)
        self.key = nn.Linear(dim, dim, bias=True)
        self.value = nn.Linear(dim, dim, bias=True)
        self.dense = nn.Linear(dim, dim, bias=True)
        self.LayerNorm = nn.LayerNorm(dim, eps=args.layer_norm_eps)

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

        # Bidirectional attention — only mask padding, no causal mask
        attn_out = scaled_dot_product_attention(q, k, v, self.scale, mask=attention_mask)

        # Reshape back to (B, T, D)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)

        # Output projection + residual + LayerNorm
        output = self.dense(attn_out)
        return self.LayerNorm(output + hidden_states)


class BertMLP(nn.Module):
    """BERT feed-forward network: dense → GELU → dense → LayerNorm."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size, bias=True)
        self.dense_out = nn.Linear(args.intermediate_size, args.hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        h = nn.gelu(self.dense(hidden_states))
        h = self.dense_out(h)
        return self.LayerNorm(h + hidden_states)


class BertLayer(nn.Module):
    """Single BERT transformer layer: attention → MLP."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = BertAttention(args)
        self.mlp = BertMLP(args)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.attention(hidden_states, attention_mask)
        return self.mlp(h)


class BertEncoder(nn.Module):
    """Stack of BERT transformer layers."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = [BertLayer(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class Model(nn.Module):
    """BERT encoder model.

    Returns hidden states (B, T, D), not logits.
    No lm_head — MLM head is separate (MLMWrapper for training).
    """

    model_category = "encoder"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = BertEmbeddings(args)
        self.encoder = BertEncoder(args)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass returning hidden states.

        Args:
            input_ids: (B, T) token IDs
            attention_mask: (B, T) binary mask (1=real, 0=padding)
            token_type_ids: (B, T) segment IDs (default: all zeros)

        Returns:
            (B, T, D) hidden states
        """
        h = self.embeddings(input_ids, token_type_ids)

        # Convert padding mask to attention mask format
        mask = create_padding_mask(attention_mask) if attention_mask is not None else None

        return self.encoder(h, mask)

    @property
    def layers(self):
        return self.encoder.layers

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Map HuggingFace BERT weight names to MLX Forge names.

        HF: bert.encoder.layer.N.attention.self.query.weight
        MLX: encoder.layers.N.attention.query.weight

        Also drops pooler weights (not used for MLM/embeddings).
        """
        sanitized = {}
        for k, v in weights.items():
            # Drop pooler
            if "pooler" in k:
                continue
            # Drop position_ids (buffer, not parameter)
            if "position_ids" in k:
                continue

            new_k = k
            # Remove "bert." prefix
            if new_k.startswith("bert."):
                new_k = new_k[5:]
            # encoder.layer.N → encoder.layers.N
            new_k = new_k.replace("encoder.layer.", "encoder.layers.")
            # attention.self.X → attention.X (remove "self." level)
            new_k = new_k.replace("attention.self.", "attention.")
            # attention.output.dense → attention.dense (output projection)
            new_k = new_k.replace("attention.output.dense", "attention.dense")
            # attention.output.LayerNorm → attention.LayerNorm
            new_k = new_k.replace("attention.output.LayerNorm", "attention.LayerNorm")
            # intermediate.dense → mlp.dense
            new_k = new_k.replace("intermediate.dense", "mlp.dense")
            # output.dense → mlp.dense_out
            new_k = new_k.replace("output.dense", "mlp.dense_out")
            # output.LayerNorm → mlp.LayerNorm
            new_k = new_k.replace("output.LayerNorm", "mlp.LayerNorm")

            sanitized[new_k] = v

        return sanitized
