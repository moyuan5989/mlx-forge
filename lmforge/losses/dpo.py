"""DPO (Direct Preference Optimization) loss functions.

V2: Per-token labels with -100 masking replaces offset-based masking.
Supports both standard DPO and reference-free DPO (SimPO).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class DPOLoss:
    """Direct Preference Optimization loss.

    Args:
        beta: Temperature parameter controlling preference strength.
              Higher beta = stronger preference signal.
        reference_free: If True, use SimPO (no reference model needed).
                       If False, standard DPO (requires reference log-probs).
    """

    def __init__(self, beta: float = 0.1, reference_free: bool = True):
        self.beta = beta
        self.reference_free = reference_free

    def __call__(self, model, chosen_ids, chosen_labels,
                 rejected_ids, rejected_labels,
                 ref_chosen_logps=None, ref_rejected_logps=None):
        """Compute DPO loss on chosen/rejected pairs.

        Args:
            model: The model being trained.
            chosen_ids: Token IDs for chosen responses, shape (B, T).
            chosen_labels: Labels for chosen, shape (B, T). -100 = masked.
            rejected_ids: Token IDs for rejected responses, shape (B, T).
            rejected_labels: Labels for rejected, shape (B, T). -100 = masked.
            ref_chosen_logps: Reference model log-probs for chosen (standard DPO only).
            ref_rejected_logps: Reference model log-probs for rejected (standard DPO only).

        Returns:
            (loss, ntoks): DPO loss and total token count.
        """
        chosen_logps, chosen_ntoks = self._sequence_logprobs(
            model, chosen_ids, chosen_labels)
        rejected_logps, rejected_ntoks = self._sequence_logprobs(
            model, rejected_ids, rejected_labels)

        if self.reference_free:
            # SimPO: use length-normalized log-probs directly
            chosen_rewards = chosen_logps / chosen_ntoks
            rejected_rewards = rejected_logps / rejected_ntoks
        else:
            # Standard DPO: log-ratio with reference model
            if ref_chosen_logps is None or ref_rejected_logps is None:
                raise ValueError(
                    "Standard DPO requires reference model log-probabilities. "
                    "Either provide ref_chosen_logps/ref_rejected_logps or "
                    "set reference_free=True (SimPO)."
                )
            chosen_rewards = chosen_logps - ref_chosen_logps
            rejected_rewards = rejected_logps - ref_rejected_logps

        # DPO objective: maximize log sigmoid of scaled reward difference
        logits = self.beta * (chosen_rewards - rejected_rewards)
        loss = -nn.log_sigmoid(logits).mean()

        ntoks = chosen_ntoks + rejected_ntoks
        return loss, ntoks

    def _sequence_logprobs(self, model, input_ids, labels):
        """Compute total log-probability for each sequence in the batch.

        Only counts tokens where labels != -100.

        Args:
            input_ids: Token IDs, shape (B, T).
            labels: Labels, shape (B, T). -100 = masked.

        Returns:
            (logps, ntoks): Per-sequence sum of log-probs, shape (B,), and token count.
        """
        logits = model(input_ids[:, :-1])
        targets = labels[:, 1:]
        mask = targets != -100

        # Per-token log-probabilities
        log_probs = -nn.losses.cross_entropy(logits, targets, reduction="none")

        # Sum log-probs per sequence (masked)
        logps = (log_probs * mask).sum(axis=1)  # (B,)
        ntoks = mask.sum()

        return logps, ntoks
