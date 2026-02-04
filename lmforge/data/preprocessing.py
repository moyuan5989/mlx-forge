"""Tokenization and template application for LMForge v0."""

from __future__ import annotations


def tokenize_dataset(
    samples: list[dict],
    tokenizer,
    fmt: str,
    *,
    mask_prompt: bool = True,
    max_seq_length: int = 2048,
) -> list[dict]:
    """Tokenize samples and compute prompt offsets.

    Returns a list of dicts with keys: "tokens" (list[int]), "offset" (int).

    Format handling:
    - chat: apply chat template, compute offset by re-encoding without last message if mask_prompt=True
    - completions: wrap in chat format, then process as chat
    - text: encode directly, offset=0 (loss on all tokens), append EOS if missing
    """
    tokenized = []

    for sample in samples:
        if fmt == "chat":
            result = _tokenize_chat(sample, tokenizer, mask_prompt, max_seq_length)
        elif fmt == "completions":
            result = _tokenize_completions(sample, tokenizer, mask_prompt, max_seq_length)
        elif fmt == "text":
            result = _tokenize_text(sample, tokenizer, max_seq_length)
        else:
            raise ValueError(f"Unknown format: {fmt}")

        tokenized.append(result)

    return tokenized


def _tokenize_chat(
    sample: dict,
    tokenizer,
    mask_prompt: bool,
    max_seq_length: int,
) -> dict:
    """Tokenize a chat format sample."""
    messages = sample["messages"]

    # Apply chat template to full conversation
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    # Extract input_ids if it's a BatchEncoding, otherwise use directly
    if hasattr(result, 'input_ids'):
        tokens = result.input_ids
    elif hasattr(result, '__getitem__') and hasattr(result[0], 'ids'):
        # Handle Encoding list
        tokens = result[0].ids
    else:
        tokens = result

    # Truncate if needed
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]

    # Compute offset for loss masking
    if mask_prompt and len(messages) > 1:
        # Re-encode without the last message to find where assistant response starts
        prompt_messages = messages[:-1]
        try:
            prompt_result = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,  # This adds the assistant prompt
            )
            # Extract input_ids if it's a BatchEncoding
            if hasattr(prompt_result, 'input_ids'):
                prompt_tokens = prompt_result.input_ids
            elif hasattr(prompt_result, '__getitem__') and hasattr(prompt_result[0], 'ids'):
                prompt_tokens = prompt_result[0].ids
            else:
                prompt_tokens = prompt_result
            offset = len(prompt_tokens)
        except Exception:
            # If re-encoding fails, fall back to offset=0 (compute loss on all tokens)
            offset = 0
    else:
        offset = 0

    return {"tokens": tokens, "offset": offset}


def _tokenize_completions(
    sample: dict,
    tokenizer,
    mask_prompt: bool,
    max_seq_length: int,
) -> dict:
    """Tokenize a completions format sample by wrapping in chat format."""
    # Wrap prompt/completion in chat messages
    messages = [
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["completion"]},
    ]

    # Create a chat-format sample and process it
    chat_sample = {"messages": messages}
    return _tokenize_chat(chat_sample, tokenizer, mask_prompt, max_seq_length)


def _tokenize_text(
    sample: dict,
    tokenizer,
    max_seq_length: int,
) -> dict:
    """Tokenize a text format sample.

    Text format: encode directly, offset=0 (loss computed on all tokens including prompt).
    Append EOS if not present.
    """
    text = sample["text"]

    # Encode text
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Append EOS if not present
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if not tokens or tokens[-1] != tokenizer.eos_token_id:
            tokens.append(tokenizer.eos_token_id)

    # Truncate if needed
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]

    # Text format: offset=0 (loss on all tokens)
    return {"tokens": tokens, "offset": 0}
