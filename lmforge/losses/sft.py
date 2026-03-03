"""SFT (Supervised Fine-Tuning) loss functions.

V2: Per-token labels with -100 masking replaces offset-based masking.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SFTLoss:
    """Standard cross-entropy SFT loss with per-token label masking."""

    def __call__(self, model, input_ids, labels):
        """Compute cross-entropy loss with per-token label masking.

        Args:
            model: The model to evaluate.
            input_ids: Token IDs array of shape (B, T).
            labels: Label IDs array of shape (B, T). -100 = masked (no loss).

        Returns:
            (loss, ntoks): Average loss per token and number of tokens used.
        """
        logits = model(input_ids[:, :-1])       # (B, T-1, V)
        targets = labels[:, 1:]                  # (B, T-1)
        mask = targets != -100

        ce = nn.losses.cross_entropy(logits, targets, reduction="none") * mask
        ntoks = mask.sum()

        return ce.sum() / ntoks, ntoks

    def packed(self, model, input_ids, labels, segment_ids):
        """Compute cross-entropy loss for packed sequences.

        Labels already encode prompt masking. segment_ids prevents
        cross-sequence loss at segment boundaries.

        Args:
            model: The model to evaluate.
            input_ids: Token IDs array of shape (B, T).
            labels: Label IDs array of shape (B, T). -100 = masked.
            segment_ids: Segment membership per token, shape (B, T). -1 = padding.

        Returns:
            (loss, ntoks): Average loss per token and number of tokens used.
        """
        logits = model(input_ids[:, :-1])
        targets = labels[:, 1:]
        seg_in = segment_ids[:, :-1]
        seg_out = segment_ids[:, 1:]

        # Mask: not -100, same segment, not padding
        mask = (targets != -100) & (seg_in == seg_out) & (seg_out >= 0)

        ce = nn.losses.cross_entropy(logits, targets, reduction="none") * mask
        ntoks = mask.sum()

        return ce.sum() / ntoks, ntoks


# Module-level convenience functions for backward compatibility
_sft_loss = SFTLoss()


def loss_fn(model, input_ids, labels):
    """Module-level SFT loss function."""
    return _sft_loss(model, input_ids, labels)


def loss_fn_packed(model, input_ids, labels, segment_ids):
    """Module-level packed SFT loss function."""
    return _sft_loss.packed(model, input_ids, labels, segment_ids)
