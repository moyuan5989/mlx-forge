"""Shared preference loss utilities and ORPO/KTO/SimPO implementations.

These three losses close the gap with mlx-tune's training method support.
All share compute_sequence_log_probs() for efficient log-prob computation.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def compute_sequence_log_probs(model, input_ids, lengths):
    """Compute per-sequence sum of log probs (shifted prediction).

    Args:
        model: Language model
        input_ids: (B, T) token IDs
        lengths: (B,) valid sequence lengths

    Returns:
        (B,) sum of log probs per sequence
    """
    logits = model(input_ids[:, :-1])           # (B, T-1, V)
    targets = input_ids[:, 1:]                   # (B, T-1)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)  # (B, T-1, V)
    # Gather target token log probs
    token_lps = mx.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)  # (B, T-1)
    # Mask by length (lengths includes BOS, so valid prediction positions = lengths - 1)
    mask = mx.arange(targets.shape[1])[None, :] < (lengths[:, None] - 1)
    return (token_lps * mask).sum(axis=-1)  # (B,)


def orpo_loss(model, chosen_ids, rejected_ids, chosen_lengths, rejected_lengths,
              beta=0.1):
    """ORPO: Odds Ratio Preference Optimization.

    No reference model needed. Combines SFT on chosen + odds ratio preference.
    L = L_SFT(chosen) + beta * L_OR
    L_SFT = -mean(log P(chosen) / |chosen|)
    L_OR  = -mean(log_sigmoid(log_odds_chosen - log_odds_rejected))

    Args:
        model: Language model
        chosen_ids: (B, T) chosen token IDs
        rejected_ids: (B, T) rejected token IDs
        chosen_lengths: (B,) chosen sequence lengths
        rejected_lengths: (B,) rejected sequence lengths
        beta: Weight for odds ratio loss

    Returns:
        (loss, ntoks): Combined loss and total token count
    """
    chosen_lps = compute_sequence_log_probs(model, chosen_ids, chosen_lengths)
    rejected_lps = compute_sequence_log_probs(model, rejected_ids, rejected_lengths)

    # Normalize by length for SFT
    chosen_avg_lps = chosen_lps / mx.maximum(chosen_lengths - 1, 1)

    # SFT loss on chosen
    sft_loss = -chosen_avg_lps.mean()

    # Odds ratio: log(odds) = log(p / (1-p)) ≈ log_prob - log(1 - exp(log_prob))
    # For numerical stability, use log_sigmoid formulation
    chosen_log_odds = chosen_lps - mx.log(1 - mx.exp(mx.minimum(chosen_lps, -1e-7)))
    rejected_log_odds = rejected_lps - mx.log(1 - mx.exp(mx.minimum(rejected_lps, -1e-7)))

    or_loss = -nn.log_sigmoid(chosen_log_odds - rejected_log_odds).mean()

    loss = sft_loss + beta * or_loss
    ntoks = mx.array((chosen_lengths.sum() + rejected_lengths.sum()).item(), dtype=mx.float32)
    return loss, ntoks


def kto_loss(model, input_ids, lengths, labels, beta=0.1):
    """KTO: Kahneman-Tversky Optimization for unpaired preferences.

    Each sample has label=1 (desirable) or label=0 (undesirable).
    No paired chosen/rejected needed.

    Args:
        model: Language model
        input_ids: (B, T) token IDs
        lengths: (B,) sequence lengths
        labels: (B,) binary labels (1=desirable, 0=undesirable)
        beta: KL penalty weight

    Returns:
        (loss, ntoks): KTO loss and total token count
    """
    log_probs = compute_sequence_log_probs(model, input_ids, lengths)

    # Reference log probs (detached current policy as baseline)
    ref_log_probs = mx.stop_gradient(log_probs)

    # Log ratio
    log_ratio = log_probs - ref_log_probs

    # Positive samples: want to increase probability
    # Negative samples: want to decrease probability
    is_positive = labels > 0.5
    is_negative = ~is_positive

    positive_loss = -nn.log_sigmoid(beta * log_ratio) * is_positive
    negative_loss = -nn.log_sigmoid(-beta * log_ratio) * is_negative

    n_pos = mx.maximum(is_positive.sum(), 1)
    n_neg = mx.maximum(is_negative.sum(), 1)

    loss = (positive_loss.sum() / n_pos + negative_loss.sum() / n_neg)
    ntoks = mx.array(lengths.sum().item(), dtype=mx.float32)
    return loss, ntoks


def simpo_loss(model, chosen_ids, rejected_ids, chosen_lengths, rejected_lengths,
               beta=2.0, gamma=0.5):
    """SimPO: Simple Preference Optimization — no reference model.

    Uses length-normalized log prob as implicit reward.
    r = log P(y|x) / |y|
    L = -mean(log_sigmoid(beta * (r_chosen - r_rejected - gamma)))

    Args:
        model: Language model
        chosen_ids: (B, T) chosen token IDs
        rejected_ids: (B, T) rejected token IDs
        chosen_lengths: (B,) chosen sequence lengths
        rejected_lengths: (B,) rejected sequence lengths
        beta: Temperature scaling
        gamma: Margin term

    Returns:
        (loss, ntoks): SimPO loss and total token count
    """
    chosen_lps = compute_sequence_log_probs(model, chosen_ids, chosen_lengths)
    rejected_lps = compute_sequence_log_probs(model, rejected_ids, rejected_lengths)

    # Length-normalized rewards
    r_chosen = chosen_lps / mx.maximum(chosen_lengths - 1, 1)
    r_rejected = rejected_lps / mx.maximum(rejected_lengths - 1, 1)

    # SimPO objective
    logits = beta * (r_chosen - r_rejected - gamma)
    loss = -nn.log_sigmoid(logits).mean()

    ntoks = mx.array((chosen_lengths.sum() + rejected_lengths.sum()).item(), dtype=mx.float32)
    return loss, ntoks
