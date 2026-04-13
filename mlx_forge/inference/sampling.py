"""Sampling strategies for text generation.

Supports:
- Greedy decoding (temperature=0)
- Temperature scaling
- Top-p (nucleus) sampling
- Top-k sampling
- Min-p sampling
- Repetition penalty
- Frequency/presence penalty
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def sample_next_token(
    logits: mx.array,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    generated_tokens: list[int] | None = None,
) -> mx.array:
    """Sample the next token from logits.

    Args:
        logits: Raw logits of shape (vocab_size,) for a single position.
        temperature: Sampling temperature. 0.0 = greedy.
        top_p: Nucleus sampling threshold. 1.0 = disabled.
        top_k: Top-k filtering. 0 = disabled.
        min_p: Min-p filtering threshold. 0.0 = disabled.
        repetition_penalty: Penalty for repeating tokens. 1.0 = disabled.
        frequency_penalty: Penalty proportional to token frequency. 0.0 = disabled.
        presence_penalty: Penalty for token presence. 0.0 = disabled.
        generated_tokens: Previously generated token IDs (for penalties).

    Returns:
        Scalar mx.array containing the sampled token ID.
    """
    # 1. Repetition penalty
    if repetition_penalty != 1.0 and generated_tokens:
        logits = _apply_repetition_penalty(logits, generated_tokens, repetition_penalty)

    # 2. Frequency/presence penalty
    if (frequency_penalty != 0.0 or presence_penalty != 0.0) and generated_tokens:
        logits = _apply_frequency_presence_penalty(
            logits, generated_tokens, frequency_penalty, presence_penalty
        )

    # 3. Greedy decoding
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)

    # 4. Temperature scaling
    temperature = max(temperature, 1e-6)
    logits = logits / temperature

    # 5. Top-k filter
    if top_k > 0:
        logits = _apply_top_k(logits, top_k)

    # 6. Min-p filter
    if min_p > 0.0:
        logits = _apply_min_p(logits, min_p)

    # 7. Top-p (nucleus) sampling
    if top_p < 1.0:
        logits = _apply_top_p(logits, top_p)

    return mx.random.categorical(logits)


def _apply_repetition_penalty(
    logits: mx.array, generated_tokens: list[int], penalty: float
) -> mx.array:
    """Apply repetition penalty to logits for previously generated tokens."""
    vocab_size = logits.shape[-1]
    clamped_tokens = [max(0, min(t, vocab_size - 1)) for t in generated_tokens]

    mask_np = np.zeros(vocab_size, dtype=np.float32)
    mask_np[clamped_tokens] = 1.0
    penalty_mask = mx.array(mask_np) > 0

    penalized_logits = mx.where(
        logits > 0,
        logits / penalty,
        logits * penalty,
    )
    return mx.where(penalty_mask, penalized_logits, logits)


def _apply_frequency_presence_penalty(
    logits: mx.array,
    generated_tokens: list[int],
    freq_penalty: float,
    pres_penalty: float,
) -> mx.array:
    """Apply frequency and presence penalties based on token counts.

    Frequency penalty: subtract freq_penalty * count(token) from logits.
    Presence penalty: subtract pres_penalty * 1(token appeared) from logits.
    """
    vocab_size = logits.shape[-1]

    # Count token frequencies
    freq_np = np.zeros(vocab_size, dtype=np.float32)
    for t in generated_tokens:
        if 0 <= t < vocab_size:
            freq_np[t] += 1.0

    # Presence is binary: appeared or not
    pres_np = (freq_np > 0).astype(np.float32)

    penalty = mx.array(freq_np * freq_penalty + pres_np * pres_penalty)
    return logits - penalty


def _apply_top_k(logits: mx.array, k: int) -> mx.array:
    """Keep only top-k tokens by logit value, set rest to -inf.

    Args:
        logits: Logits of shape (vocab_size,)
        k: Number of top tokens to keep.

    Returns:
        Filtered logits with tokens outside top-k set to -inf.
    """
    vocab_size = logits.shape[-1]
    k = min(k, vocab_size)

    if k >= vocab_size:
        return logits

    # Find the k-th largest value as threshold
    # argpartition puts the top-k in the last k positions (unsorted)
    top_k_indices = mx.argpartition(-logits, kth=k - 1)
    # The k-th value (0-indexed at k-1) is the cutoff
    kth_val = logits[top_k_indices[k - 1]]

    # Mask everything below the threshold
    mask = logits < kth_val
    return mx.where(mask, mx.array(float("-inf")), logits)


def _apply_min_p(logits: mx.array, min_p: float) -> mx.array:
    """Filter tokens with probability below min_p * max_prob.

    Args:
        logits: Logits of shape (vocab_size,)
        min_p: Minimum probability ratio threshold.

    Returns:
        Filtered logits with low-probability tokens set to -inf.
    """
    probs = mx.softmax(logits, axis=-1)
    max_prob = mx.max(probs)
    threshold = min_p * max_prob

    mask = probs < threshold
    filtered = mx.where(mask, mx.array(float("-inf")), logits)

    # Fallback: if all filtered out, keep the top token
    all_neg_inf = mx.all(filtered == float("-inf"))
    if all_neg_inf.item():
        top_idx = mx.argmax(probs)
        # Rebuild with only top token via numpy (MLX scatter workaround)
        mask_np = np.full(logits.shape[-1], float("-inf"), dtype=np.float32)
        mask_np[top_idx.item()] = logits[top_idx].item()
        return mx.array(mask_np)

    return filtered


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

    # Fallback: if all tokens are -inf (top_p too small), keep the top-1 token
    all_neg_inf = mx.all(sorted_logits == float("-inf"))
    if all_neg_inf.item():
        sorted_logits = mx.full_like(sorted_logits, float("-inf"))
        sorted_logits = sorted_logits.at[0].add(float("inf"))  # top-1 in sorted order

    # Scatter back to original order using inverse permutation (gather, not scatter)
    inverse_indices = mx.argsort(sorted_indices)
    return sorted_logits[inverse_indices]
