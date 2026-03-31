"""Seq2seq loss for encoder-decoder models.

Decoder-side shifting: logits from decoder_input_ids[:, :-1] compared
against decoder_labels[:, 1:]. This is teacher-forcing with left-shift.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class Seq2SeqLoss:
    """Cross-entropy loss for sequence-to-sequence training."""

    def __call__(
        self,
        model,
        encoder_input_ids,
        decoder_input_ids,
        decoder_labels,
        encoder_attention_mask=None,
    ):
        """Compute seq2seq cross-entropy loss.

        Args:
            model: Encoder-decoder model.
            encoder_input_ids: (B, T_enc) source token IDs.
            decoder_input_ids: (B, T_dec) target token IDs (teacher forcing input).
            decoder_labels: (B, T_dec) target labels, -100 = masked.
            encoder_attention_mask: (B, T_enc) padding mask.

        Returns:
            (loss, ntoks): Average cross-entropy, token count.
        """
        logits = model(
            encoder_input_ids,
            decoder_input_ids[:, :-1],
            encoder_attention_mask,
        )  # (B, T_dec-1, V)

        targets = decoder_labels[:, 1:]  # (B, T_dec-1)
        mask = targets != -100

        ce = nn.losses.cross_entropy(logits, targets, reduction="none") * mask
        ntoks = mask.sum()

        return ce.sum() / mx.maximum(ntoks, 1), ntoks
