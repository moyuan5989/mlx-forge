"""GRPO (Group Relative Policy Optimization) loss for MLX Forge.

Implements PPO-style clipped surrogate objective with KL penalty,
as described in the DeepSeek-R1 paper.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class GRPOLoss:
    """GRPO loss: clipped surrogate with KL divergence penalty.

    loss = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage) + beta * KL
    where ratio = exp(policy_logprob - ref_logprob)
    """

    def __init__(self, beta: float = 0.1, clip_range: float = 0.2):
        self.beta = beta
        self.clip_range = clip_range

    def __call__(
        self,
        policy_logprobs: mx.array,
        ref_logprobs: mx.array,
        advantages: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Compute GRPO loss.

        Args:
            policy_logprobs: Log probs from current policy, shape (B, T)
            ref_logprobs: Log probs from reference policy, shape (B, T)
            advantages: Normalized advantages, shape (B,) or (B, 1)
            mask: Optional mask for valid tokens, shape (B, T)

        Returns:
            Scalar loss.
        """
        # Importance ratio
        ratio = mx.exp(policy_logprobs - ref_logprobs)

        # Expand advantages to match token dimension
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        # Clipped surrogate
        clipped_ratio = mx.clip(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        surrogate = mx.minimum(ratio * advantages, clipped_ratio * advantages)

        # KL divergence penalty (approximation: policy_lp - ref_lp)
        kl = policy_logprobs - ref_logprobs

        # Combined loss
        loss = -surrogate + self.beta * kl

        if mask is not None:
            loss = loss * mask
            return loss.sum() / mx.maximum(mask.sum(), 1.0)
        else:
            return loss.mean()


def compute_log_probs(
    model: nn.Module,
    input_ids: mx.array,
    labels: mx.array,
) -> mx.array:
    """Compute per-token log probabilities.

    Args:
        model: Language model
        input_ids: Input token IDs, shape (B, T)
        labels: Target token IDs, shape (B, T)

    Returns:
        Log probs for each position, shape (B, T)
    """
    logits = model(input_ids)  # (B, T, V)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Gather log probs for target tokens
    # labels shape: (B, T), need to gather from (B, T, V)
    batch_size, seq_len = labels.shape
    token_log_probs = mx.zeros((batch_size, seq_len))

    for b in range(batch_size):
        for t in range(seq_len):
            token_log_probs = token_log_probs.at[b, t].add(
                log_probs[b, t, labels[b, t]] - token_log_probs[b, t]
            )

    return token_log_probs


def compute_log_probs_fast(
    model: nn.Module,
    input_ids: mx.array,
    labels: mx.array,
) -> mx.array:
    """Compute per-token log probabilities (vectorized).

    Args:
        model: Language model
        input_ids: Input token IDs, shape (B, T)
        labels: Target token IDs, shape (B, T)

    Returns:
        Log probs for each position, shape (B, T-1) (shifted)
    """
    logits = model(input_ids)  # (B, T, V)
    # Shift: predict next token
    shift_logits = logits[:, :-1, :]  # (B, T-1, V)
    shift_labels = labels[:, 1:]       # (B, T-1)

    log_probs = mx.log_softmax(shift_logits, axis=-1)  # (B, T-1, V)

    # Gather using advanced indexing
    batch_size, seq_len = shift_labels.shape
    batch_idx = mx.repeat(mx.arange(batch_size)[:, None], seq_len, axis=1)
    seq_idx = mx.repeat(mx.arange(seq_len)[None, :], batch_size, axis=0)

    token_log_probs = log_probs[batch_idx, seq_idx, shift_labels]

    return token_log_probs


def compute_group_advantages(rewards: mx.array) -> mx.array:
    """Compute group-normalized advantages from rewards.

    advantages = (reward - mean(rewards)) / (std(rewards) + eps)

    Args:
        rewards: Reward scores, shape (G,) for G generations

    Returns:
        Normalized advantages, shape (G,)
    """
    mean = mx.mean(rewards)
    std = mx.sqrt(mx.mean((rewards - mean) ** 2))
    return (rewards - mean) / (std + 1e-8)
