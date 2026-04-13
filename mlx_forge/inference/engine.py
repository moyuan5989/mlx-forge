"""Generation engine for text generation with optional LoRA adapters.

Provides:
- load_for_inference(): Load model + optional adapter, fuse weights
- generate_tokens(): Autoregressive token generation (yields token IDs)
- generate_steps(): Enhanced generation yielding StepResult with logprobs/metrics
- generate(): High-level generation returning GenerationResult
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mlx.core as mx

from mlx_forge.inference.cache import make_cache
from mlx_forge.inference.logprobs import TokenLogprobResult, compute_logprobs
from mlx_forge.inference.metrics import GenerationMetrics, MetricsTracker
from mlx_forge.inference.sampling import sample_next_token


@dataclass
class StepResult:
    """Result from a single generation step."""

    token_id: int
    logprob_result: TokenLogprobResult | None = None


@dataclass
class GenerationResult:
    """Result of a text generation call."""

    text: str
    prompt: str
    num_tokens: int
    tokens_per_second: float
    finish_reason: str  # "stop" (EOS), "length" (max_tokens)
    logprobs_list: list[TokenLogprobResult] | None = None
    metrics: GenerationMetrics | None = None


def load_for_inference(
    model_path: str,
    adapter_path: str | None = None,
    trust_remote_code: bool = False,
):
    """Load a model and optionally apply LoRA adapter weights for inference.

    Args:
        model_path: HuggingFace model ID or local path.
        adapter_path: Path to checkpoint directory containing adapters.safetensors.
        trust_remote_code: Passed to tokenizer loading.

    Returns:
        Tuple of (model, tokenizer).
    """
    from mlx_forge.models.loader import load_model
    from mlx_forge.models.resolve import resolve_model

    # Resolve model path (HF ID → local)
    resolved = resolve_model(model_path, trust_remote_code=trust_remote_code)
    model, tokenizer = load_model(
        resolved.local_path, trust_remote_code=trust_remote_code
    )

    # Load adapter weights if provided
    if adapter_path:
        adapter_path = Path(adapter_path).expanduser()
        adapter_file = adapter_path / "adapters.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(
                f"No adapters.safetensors in {adapter_path}. "
                f"Provide a checkpoint directory from a training run."
            )
        adapter_weights = mx.load(str(adapter_file))
        model.load_weights(list(adapter_weights.items()), strict=False)

    model.eval()
    mx.eval(model.parameters())
    return model, tokenizer


def _get_num_layers(model) -> int:
    """Get the number of transformer layers in a model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    elif hasattr(model, "layers"):
        return len(model.layers)
    else:
        raise ValueError("Cannot determine number of layers in model")


def _make_model_cache(model, max_size: int, num_keep: int = 0):
    """Create cache for model, preferring model-level method.

    Args:
        model: The model instance.
        max_size: Maximum cache size in tokens.
        num_keep: Tokens to preserve at start on rotation (0 = use KVCache).
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = _get_num_layers(model)

    if num_keep > 0 and max_size > 0:
        from mlx_forge.inference.rotating_cache import make_rotating_cache

        return make_rotating_cache(num_layers, max_size=max_size, num_keep=num_keep)

    return make_cache(num_layers, max_size=max_size)


def generate_tokens(
    model,
    prompt_tokens: list[int],
    tokenizer,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    max_tokens: int = 512,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
):
    """Generate tokens autoregressively. Yields token IDs one at a time.

    Args:
        model: The loaded model (with or without LoRA weights).
        prompt_tokens: Tokenized prompt as list of int IDs.
        tokenizer: Tokenizer instance (for EOS detection).
        temperature: Sampling temperature. 0.0 = greedy.
        top_p: Nucleus sampling threshold.
        top_k: Top-k filtering. 0 = disabled.
        min_p: Min-p filtering threshold. 0.0 = disabled.
        max_tokens: Maximum number of tokens to generate.
        repetition_penalty: Penalty for repeating tokens.
        frequency_penalty: Frequency penalty. 0.0 = disabled.
        presence_penalty: Presence penalty. 0.0 = disabled.
        seed: Optional RNG seed for reproducible generation.

    Yields:
        Token IDs (int) one at a time.
    """
    for step in generate_steps(
        model,
        prompt_tokens,
        tokenizer,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        logprobs=False,
    ):
        yield step.token_id


def generate_steps(
    model,
    prompt_tokens: list[int],
    tokenizer,
    *,
    cache: list | None = None,
    all_token_history: list[int] | None = None,
    context_length: int = 0,
    num_keep: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    max_tokens: int = 512,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
    logprobs: bool = False,
    top_logprobs: int = 5,
) -> Iterator[StepResult]:
    """Enhanced generation yielding StepResult with optional logprobs.

    Args:
        model: The loaded model.
        prompt_tokens: Tokenized prompt as list of int IDs. If cache is
            provided, these should be only the NEW tokens to prefill
            (the cache already contains earlier context).
        tokenizer: Tokenizer instance (for EOS detection).
        cache: Optional pre-existing KV cache (for multi-turn reuse).
        all_token_history: Full token history for repetition/frequency
            penalty tracking. If None, uses prompt_tokens.
        context_length: Fixed context window size. 0 = auto (prompt + max_tokens).
            When set, uses RotatingKVCache if num_keep > 0.
        num_keep: Tokens to preserve at start of context on rotation.
            Only used when context_length > 0.
        temperature: Sampling temperature. 0.0 = greedy.
        top_p: Nucleus sampling threshold.
        top_k: Top-k filtering. 0 = disabled.
        min_p: Min-p filtering threshold. 0.0 = disabled.
        max_tokens: Maximum number of tokens to generate.
        repetition_penalty: Penalty for repeating tokens.
        frequency_penalty: Frequency penalty. 0.0 = disabled.
        presence_penalty: Presence penalty. 0.0 = disabled.
        seed: Optional RNG seed for reproducible generation.
        logprobs: Whether to compute logprobs for each token.
        top_logprobs: Number of top alternative logprobs to include.

    Yields:
        StepResult with token_id and optional logprob_result.
    """
    if seed is not None:
        mx.random.seed(seed)

    if cache is None:
        if context_length > 0:
            cache_max_size = context_length
        else:
            cache_max_size = len(prompt_tokens) + max_tokens
        cache = _make_model_cache(model, max_size=cache_max_size, num_keep=num_keep)

    # Prefill: process prompt tokens (may be delta if cache is reused)
    tokens = mx.array(prompt_tokens)[None]  # (1, T)
    model_logits = model(tokens, cache=cache)

    # For penalties: track full history (not just prefilled tokens)
    generated = list(all_token_history or prompt_tokens)
    last_logits = model_logits[0, -1, :]
    next_token = sample_next_token(
        last_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        generated_tokens=generated,
    )
    mx.eval(next_token)

    for _ in range(max_tokens):
        token_id = next_token.item()

        # Check for EOS
        if token_id == tokenizer.eos_token_id:
            return

        # Compute logprobs if requested
        logprob_result = None
        if logprobs:
            logprob_result = compute_logprobs(
                last_logits, token_id, tokenizer, top_n=top_logprobs
            )

        yield StepResult(token_id=token_id, logprob_result=logprob_result)
        generated.append(token_id)

        # Decode next token
        next_input = next_token.reshape(1, 1)
        model_logits = model(next_input, cache=cache)
        last_logits = model_logits[0, -1, :]
        next_token = sample_next_token(
            last_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            generated_tokens=generated,
        )
        mx.eval(next_token)


def generate_seq2seq_tokens(
    model,
    input_tokens: list[int],
    tokenizer,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    seed: int | None = None,
):
    """Generate tokens from an encoder-decoder model. Yields token IDs.

    Two-phase generation:
    1. Encode the input sequence once.
    2. Decode autoregressively using cached encoder hidden states.

    Args:
        model: Encoder-decoder model (model_category == "encoder_decoder").
        input_tokens: Tokenized source sequence.
        tokenizer: Tokenizer instance (for EOS detection).
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum number of tokens to generate.
        seed: Optional RNG seed.

    Yields:
        Token IDs (int) one at a time.
    """
    if seed is not None:
        mx.random.seed(seed)

    # Phase 1: Encode input
    enc_ids = mx.array(input_tokens)[None]  # (1, T_enc)
    encoder_hidden = model.encode(enc_ids)

    # Phase 2: Decode autoregressively
    cache = model.make_cache()

    # Start with decoder_start_token_id
    decoder_start_id = getattr(model.args, "decoder_start_token_id", 0)
    dec_token = mx.array([[decoder_start_id]])

    logits = model.decode(dec_token, encoder_hidden, cache=cache)
    next_token = sample_next_token(
        logits[0, -1, :],
        temperature=temperature,
        top_p=top_p,
    )
    mx.eval(next_token)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(model.args, "eos_token_id", 1)

    for _ in range(max_tokens):
        token_id = next_token.item()

        if token_id == eos_id:
            return

        yield token_id

        dec_token = next_token.reshape(1, 1)
        logits = model.decode(dec_token, encoder_hidden, cache=cache)
        next_token = sample_next_token(
            logits[0, -1, :],
            temperature=temperature,
            top_p=top_p,
        )
        mx.eval(next_token)


def generate(
    model,
    tokenizer,
    prompt: str | None = None,
    messages: list[dict] | None = None,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    max_tokens: int = 512,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
    logprobs: bool = False,
    top_logprobs: int = 5,
) -> GenerationResult:
    """Generate text from a prompt or chat messages.

    Args:
        model: Loaded model instance.
        tokenizer: Tokenizer instance.
        prompt: Raw text prompt (mutually exclusive with messages).
        messages: Chat messages list (mutually exclusive with prompt).
        temperature: Sampling temperature. 0.0 = greedy.
        top_p: Nucleus sampling threshold.
        top_k: Top-k filtering. 0 = disabled.
        min_p: Min-p filtering threshold. 0.0 = disabled.
        max_tokens: Maximum number of tokens to generate.
        repetition_penalty: Penalty for repeating tokens.
        frequency_penalty: Frequency penalty. 0.0 = disabled.
        presence_penalty: Presence penalty. 0.0 = disabled.
        seed: Optional RNG seed for reproducible generation.
        logprobs: Whether to compute logprobs for each token.
        top_logprobs: Number of top alternative logprobs to include.

    Returns:
        GenerationResult with generated text, stats, and finish reason.
    """
    if prompt is None and messages is None:
        raise ValueError("Must provide either 'prompt' or 'messages'")
    if prompt is not None and messages is not None:
        raise ValueError("Provide 'prompt' or 'messages', not both")

    # Tokenize
    if messages is not None:
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        if not isinstance(prompt_tokens, list):
            prompt_tokens = prompt_tokens["input_ids"]
        prompt_text = tokenizer.decode(prompt_tokens)
    else:
        prompt_tokens = tokenizer.encode(prompt)
        prompt_text = prompt

    # Track metrics
    tracker = MetricsTracker(num_prompt_tokens=len(prompt_tokens))

    # Generate
    generated_ids = []
    logprobs_list = [] if logprobs else None
    finish_reason = "length"
    first_token = True

    for step in generate_steps(
        model,
        prompt_tokens,
        tokenizer,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    ):
        if first_token:
            tracker.mark_prefill_done()
            first_token = False

        tracker.mark_token()
        generated_ids.append(step.token_id)
        if logprobs and step.logprob_result is not None:
            logprobs_list.append(step.logprob_result)

    # If generate_steps returned early (EOS), it's a stop
    if len(generated_ids) < max_tokens:
        finish_reason = "stop"

    metrics = tracker.finish()
    num_tokens = len(generated_ids)
    tokens_per_second = (
        num_tokens / (metrics.total_time_ms / 1000) if metrics.total_time_ms > 0 else 0.0
    )
    text = tokenizer.decode(generated_ids)

    return GenerationResult(
        text=text,
        prompt=prompt_text,
        num_tokens=num_tokens,
        tokens_per_second=tokens_per_second,
        finish_reason=finish_reason,
        logprobs_list=logprobs_list,
        metrics=metrics,
    )
