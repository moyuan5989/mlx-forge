"""Logprobs computation for token-level probability analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class TokenLogprob:
    """A single token with its log probability."""

    token: str
    token_id: int
    logprob: float


@dataclass
class TokenLogprobResult:
    """Log probability result for a generated token, including top-N alternatives."""

    token: str
    token_id: int
    logprob: float
    top_logprobs: list[TokenLogprob] = field(default_factory=list)


def compute_logprobs(
    logits: mx.array,
    selected_token_id: int,
    tokenizer,
    top_n: int = 5,
) -> TokenLogprobResult:
    """Compute log-softmax and extract selected token's logprob + top-N alternatives.

    Args:
        logits: Raw logits of shape (vocab_size,) for a single position.
        selected_token_id: The token that was actually sampled.
        tokenizer: Tokenizer for decoding token IDs to strings.
        top_n: Number of top alternative tokens to include.

    Returns:
        TokenLogprobResult with the selected token's logprob and top-N alternatives.
    """
    # Compute log-softmax for numerical stability
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Get selected token's logprob
    selected_logprob = log_probs[selected_token_id].item()
    selected_token_str = tokenizer.decode([selected_token_id])

    # Get top-N tokens by log probability
    top_n = min(top_n, logits.shape[-1])
    top_indices = mx.argpartition(-log_probs, kth=top_n - 1)[:top_n]
    top_log_vals = log_probs[top_indices]

    # Sort by descending logprob
    sort_order = mx.argsort(-top_log_vals)
    top_indices = top_indices[sort_order]
    top_log_vals = top_log_vals[sort_order]

    top_logprobs = []
    for i in range(top_n):
        tid = top_indices[i].item()
        lp = top_log_vals[i].item()
        tok_str = tokenizer.decode([tid])
        top_logprobs.append(TokenLogprob(token=tok_str, token_id=tid, logprob=lp))

    return TokenLogprobResult(
        token=selected_token_str,
        token_id=selected_token_id,
        logprob=selected_logprob,
        top_logprobs=top_logprobs,
    )
