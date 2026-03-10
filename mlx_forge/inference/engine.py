"""Generation engine for text generation with optional LoRA adapters.

Provides:
- load_for_inference(): Load model + optional adapter, fuse weights
- generate_tokens(): Autoregressive token generation (yields token IDs)
- generate(): High-level generation returning GenerationResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mlx_forge.inference.cache import make_cache
from mlx_forge.inference.sampling import sample_next_token


@dataclass
class GenerationResult:
    """Result of a text generation call."""

    text: str
    prompt: str
    num_tokens: int
    tokens_per_second: float
    finish_reason: str  # "stop" (EOS), "length" (max_tokens)


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


def generate_tokens(
    model,
    prompt_tokens: list[int],
    tokenizer,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
):
    """Generate tokens autoregressively. Yields token IDs one at a time.

    Args:
        model: The loaded model (with or without LoRA weights).
        prompt_tokens: Tokenized prompt as list of int IDs.
        tokenizer: Tokenizer instance (for EOS detection).
        temperature: Sampling temperature. 0.0 = greedy.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum number of tokens to generate.
        repetition_penalty: Penalty for repeating tokens.
        seed: Optional RNG seed for reproducible generation.

    Yields:
        Token IDs (int) one at a time.
    """
    if seed is not None:
        mx.random.seed(seed)

    # Pre-compute max cache size: prompt + generated tokens
    cache_max_size = len(prompt_tokens) + max_tokens

    # Create cache — prefer model-level method for hybrid architectures
    if hasattr(model, "make_cache"):
        cache = model.make_cache()
    else:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            num_layers = len(model.model.layers)
        elif hasattr(model, "layers"):
            num_layers = len(model.layers)
        else:
            raise ValueError("Cannot determine number of layers in model")
        cache = make_cache(num_layers, max_size=cache_max_size)

    # Prefill: process entire prompt at once
    tokens = mx.array(prompt_tokens)[None]  # (1, T)
    logits = model(tokens, cache=cache)

    # Sample first token from last position's logits
    generated = list(prompt_tokens)
    next_token = sample_next_token(
        logits[0, -1, :],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        generated_tokens=generated,
    )
    mx.eval(next_token)

    for _ in range(max_tokens):
        token_id = next_token.item()

        # Check for EOS
        if token_id == tokenizer.eos_token_id:
            return

        yield token_id
        generated.append(token_id)

        # Decode next token
        next_input = next_token.reshape(1, 1)
        logits = model(next_input, cache=cache)
        next_token = sample_next_token(
            logits[0, -1, :],
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=generated,
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
    max_tokens: int = 512,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
) -> GenerationResult:
    """Generate text from a prompt or chat messages.

    Args:
        model: Loaded model instance.
        tokenizer: Tokenizer instance.
        prompt: Raw text prompt (mutually exclusive with messages).
        messages: Chat messages list (mutually exclusive with prompt).
        temperature: Sampling temperature. 0.0 = greedy.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum number of tokens to generate.
        repetition_penalty: Penalty for repeating tokens.
        seed: Optional RNG seed for reproducible generation.

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

    # Generate
    t0 = time.perf_counter()
    generated_ids = []
    finish_reason = "length"

    for token_id in generate_tokens(
        model,
        prompt_tokens,
        tokenizer,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        seed=seed,
    ):
        generated_ids.append(token_id)

    # If generate_tokens returned early (EOS), it's a stop
    if len(generated_ids) < max_tokens:
        finish_reason = "stop"

    elapsed = time.perf_counter() - t0
    num_tokens = len(generated_ids)
    tokens_per_second = num_tokens / elapsed if elapsed > 0 else 0.0
    text = tokenizer.decode(generated_ids)

    return GenerationResult(
        text=text,
        prompt=prompt_text,
        num_tokens=num_tokens,
        tokens_per_second=tokens_per_second,
        finish_reason=finish_reason,
    )
