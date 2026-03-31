"""Masked Language Modeling (MLM) loss for encoder models.

Key difference from SFT: NO label shifting. MLM predicts tokens at the
same positions (masked positions only, indicated by labels != -100).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MLMLoss:
    """MLM cross-entropy loss with per-token masking (no shift)."""

    def __call__(self, model, input_ids, labels, attention_mask=None):
        """Compute MLM loss.

        Args:
            model: MLMWrapper (encoder + mlm_head).
            input_ids: (B, T) token IDs with [MASK] tokens.
            labels: (B, T) original token IDs at masked positions, -100 elsewhere.
            attention_mask: (B, T) binary mask for padding.

        Returns:
            (loss, ntoks): Average cross-entropy over masked tokens.
        """
        logits = model(input_ids, attention_mask)  # (B, T, V)
        mask = labels != -100

        ce = nn.losses.cross_entropy(logits, labels, reduction="none") * mask
        ntoks = mask.sum()

        return ce.sum() / mx.maximum(ntoks, 1), ntoks


class MLMWrapper(nn.Module):
    """Wraps encoder + MLM head so nn.value_and_grad() computes gradients for both.

    The MLM head projects hidden states back to vocabulary logits.
    """

    def __init__(self, encoder: nn.Module, mlm_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.mlm_head = mlm_head

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass: encoder → mlm_head → logits.

        Returns: (B, T, V) logits.
        """
        hidden = self.encoder(input_ids, attention_mask)
        return self.mlm_head(hidden)

    @property
    def layers(self):
        return self.encoder.layers


class MLMHead(nn.Module):
    """MLM prediction head: dense → activation → LayerNorm → projection."""

    def __init__(self, hidden_size: int, vocab_size: int, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        h = nn.gelu(self.dense(hidden_states))
        h = self.LayerNorm(h)
        return self.decoder(h)
