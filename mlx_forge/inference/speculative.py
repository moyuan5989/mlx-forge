"""Speculative decoding for faster inference.

Uses a smaller draft model to generate candidate tokens, then verifies
them with the main model in a single forward pass. Accepted tokens are
"free" — yielding 1.5-2x speedup for compatible model pairs.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_forge.inference.cache import make_cache
from mlx_forge.inference.sampling import sample_next_token


def speculative_generate_tokens(
    model,
    draft_model,
    prompt_tokens: list[int],
    tokenizer,
    *,
    num_draft: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
):
    """Generate with speculative decoding.

    Yields (token_id, from_draft: bool) tuples.

    Args:
        model: Main (verifier) model
        draft_model: Smaller draft model (same tokenizer)
        prompt_tokens: Tokenized prompt
        tokenizer: Tokenizer for EOS detection
        num_draft: Number of draft tokens per verification step
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        max_tokens: Maximum tokens to generate
        repetition_penalty: Repetition penalty
        seed: Optional RNG seed
    """
    if seed is not None:
        mx.random.seed(seed)

    cache_max = len(prompt_tokens) + max_tokens
    model_cache = _make_model_cache(model, cache_max)
    draft_cache = _make_model_cache(draft_model, cache_max)

    tokens = mx.array(prompt_tokens)[None]

    # Prefill both models
    model_logits = model(tokens, cache=model_cache)
    draft_model(tokens, cache=draft_cache)
    mx.eval(model_logits)

    generated = list(prompt_tokens)
    total_generated = 0

    # Sample first token from main model
    y = sample_next_token(
        model_logits[0, -1, :],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        generated_tokens=generated,
    )
    mx.eval(y)

    while total_generated < max_tokens:
        token_id = y.item()
        if token_id == tokenizer.eos_token_id:
            return

        yield token_id, False
        generated.append(token_id)
        total_generated += 1

        if total_generated >= max_tokens:
            return

        # Draft phase: generate num_draft candidate tokens
        draft_tokens = []
        draft_y = y.reshape(1, 1)

        for _ in range(num_draft):
            draft_logits = draft_model(draft_y, cache=draft_cache)
            draft_next = sample_next_token(
                draft_logits[0, -1, :],
                temperature=temperature,
                top_p=top_p,
            )
            mx.eval(draft_next)
            draft_token = draft_next.item()
            draft_tokens.append(draft_token)
            if draft_token == tokenizer.eos_token_id:
                break
            draft_y = draft_next.reshape(1, 1)

        if not draft_tokens:
            # Draft produced nothing, fall back to normal decode
            next_input = y.reshape(1, 1)
            model_logits = model(next_input, cache=model_cache)
            y = sample_next_token(
                model_logits[0, -1, :],
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated,
            )
            mx.eval(y)
            continue

        # Verify phase: run main model on draft tokens
        verify_input = mx.array([[y.item()] + draft_tokens])
        verify_logits = model(verify_input, cache=model_cache)
        mx.eval(verify_logits)

        # Accept/reject token by token
        n_accepted = 0
        for i, draft_token in enumerate(draft_tokens):
            main_token = sample_next_token(
                verify_logits[0, i, :],
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated,
            )
            mx.eval(main_token)

            if main_token.item() == draft_token:
                # Accept draft token
                yield draft_token, True
                generated.append(draft_token)
                total_generated += 1
                n_accepted += 1

                if draft_token == tokenizer.eos_token_id:
                    return
                if total_generated >= max_tokens:
                    return
            else:
                # Reject: use main model's token instead
                y = main_token
                break
        else:
            # All draft tokens accepted, sample next from last verify position
            y = sample_next_token(
                verify_logits[0, len(draft_tokens), :],
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated,
            )
            mx.eval(y)

        # Rewind caches for rejected tokens
        n_rejected = len(draft_tokens) - n_accepted
        if n_rejected > 0:
            _trim_cache(model_cache, n_rejected)
            _trim_cache(draft_cache, max(n_rejected - 1, 0))


def _make_model_cache(model, max_size):
    """Create cache for a model."""
    if hasattr(model, "make_cache"):
        return model.make_cache()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "layers"):
        num_layers = len(model.layers)
    else:
        raise ValueError("Cannot determine number of layers")
    return make_cache(num_layers, max_size=max_size)


def _trim_cache(cache, n):
    """Remove last n tokens from all cache layers."""
    if n <= 0:
        return
    for c in cache:
        if hasattr(c, "trim"):
            c.trim(n)
        elif hasattr(c, "offset"):
            c.offset = max(0, c.offset - n)
