"""Sampling strategies for text generation.

Supports:
- Greedy decoding (temperature=0)
- Temperature scaling
- Top-p (nucleus) sampling
- Repetition penalty
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def sample_next_token(
    logits: mx.array,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    generated_tokens: list[int] | None = None,
) -> mx.array:
    """Sample the next token from logits.

    Args:
        logits: Raw logits of shape (vocab_size,) for a single position.
        temperature: Sampling temperature. 0.0 = greedy.
        top_p: Nucleus sampling threshold. 1.0 = disabled.
        repetition_penalty: Penalty for repeating tokens. 1.0 = disabled.
        generated_tokens: Previously generated token IDs (for repetition penalty).

    Returns:
        Scalar mx.array containing the sampled token ID.
    """
    # Apply repetition penalty
    if repetition_penalty != 1.0 and generated_tokens:
        # Build boolean mask on CPU (MLX lacks scatter)
        mask_np = np.zeros(logits.shape[-1], dtype=np.float32)
        mask_np[generated_tokens] = 1.0
        penalty_mask = mx.array(mask_np) > 0

        # Compute penalized logits for ALL positions, then select via mask
        penalized_logits = mx.where(
            logits > 0,
            logits / repetition_penalty,
            logits * repetition_penalty,
        )
        logits = mx.where(penalty_mask, penalized_logits, logits)

    # Greedy decoding
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)

    # Apply temperature
    logits = logits / temperature

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        logits = _apply_top_p(logits, top_p)

    return mx.random.categorical(logits)


def _apply_top_p(logits: mx.array, top_p: float) -> mx.array:
    """Filter logits to keep only tokens within the top-p probability mass.

    Args:
        logits: Logits of shape (vocab_size,)
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits with tokens outside top-p set to -inf.
    """
    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(probs, axis=-1)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Create mask: keep tokens where cumulative prob (before this token) < top_p
    cutoff_mask = (cumulative_probs - sorted_probs) >= top_p

    # Set filtered logits to -inf in sorted order
    sorted_logits = logits[sorted_indices]
    sorted_logits = mx.where(cutoff_mask, mx.array(float("-inf")), sorted_logits)

    # Scatter back to original order using inverse permutation (gather, not scatter)
    inverse_indices = mx.argsort(sorted_indices)
    return sorted_logits[inverse_indices]
