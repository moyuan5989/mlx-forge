"""Tokenization with per-token labels for MLX Forge V2.

Returns {"input_ids": list[int], "labels": list[int]} where labels use
-100 to mask tokens that should not contribute to loss. This enables
training on ALL assistant turns in multi-turn conversations.
"""

from __future__ import annotations


def tokenize_single(
    sample: dict,
    tokenizer,
    *,
    max_seq_length: int = 2048,
    mask_prompt: bool = True,
) -> dict | None:
    """Tokenize a single sample, auto-detecting format.

    Returns dict with "input_ids" and "labels" keys, or None on failure.
    Used by streaming data pipeline for on-the-fly tokenization.
    """
    from mlx_forge.data.formats import detect_format

    try:
        fmt = detect_format([sample])
    except ValueError:
        return None

    if fmt == "chat":
        return _tokenize_chat(sample, tokenizer, mask_prompt, max_seq_length)
    elif fmt == "completions":
        return _tokenize_completions(sample, tokenizer, mask_prompt, max_seq_length)
    elif fmt == "text" or fmt == "kto":
        return _tokenize_text(sample, tokenizer, max_seq_length)
    elif fmt == "preference":
        return _tokenize_preference(sample, tokenizer, mask_prompt, max_seq_length)
    return None


def tokenize_dataset(
    samples: list[dict],
    tokenizer,
    fmt: str,
    *,
    mask_prompt: bool = True,
    max_seq_length: int = 2048,
) -> list[dict]:
    """Tokenize samples with per-token labels.

    Returns a list of dicts with keys:
    - For chat/completions/text: "input_ids" (list[int]), "labels" (list[int])
    - For preference: "chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"

    Labels use -100 to mark tokens that should not contribute to loss.
    """
    tokenized = []

    for sample in samples:
        if fmt == "chat":
            result = _tokenize_chat(sample, tokenizer, mask_prompt, max_seq_length)
        elif fmt == "completions":
            result = _tokenize_completions(sample, tokenizer, mask_prompt, max_seq_length)
        elif fmt == "text":
            result = _tokenize_text(sample, tokenizer, max_seq_length)
        elif fmt == "preference":
            result = _tokenize_preference(sample, tokenizer, mask_prompt, max_seq_length)
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
    """Tokenize a chat format sample with per-token labels.

    Key improvement over V1: trains on ALL assistant turns, not just the last one.
    Each assistant turn's tokens get real labels; user/system turns get -100.
    """
    messages = sample["messages"]

    # Tokenize the full conversation
    full_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(full_tokens, 'input_ids'):
        full_tokens = full_tokens.input_ids
    elif hasattr(full_tokens, '__getitem__') and hasattr(full_tokens[0], 'ids'):
        full_tokens = full_tokens[0].ids

    input_ids = list(full_tokens)

    if not mask_prompt:
        # No masking: train on all tokens
        labels = list(input_ids)
    else:
        # Build labels by finding assistant turn boundaries
        labels = [-100] * len(input_ids)

        # Pre-compute cumulative token lengths to find boundaries in O(N)
        # instead of O(N²) re-tokenizations per assistant turn.
        # cum_lens[i] = len(tokenize(messages[:i], add_generation_prompt=False))
        cum_lens = [0]  # cum_lens[0] = 0 tokens for empty prefix
        for i in range(len(messages)):
            try:
                toks_up_to = tokenizer.apply_chat_template(
                    messages[:i + 1],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                if hasattr(toks_up_to, 'input_ids'):
                    toks_up_to = toks_up_to.input_ids
                elif hasattr(toks_up_to, '__getitem__') and hasattr(toks_up_to[0], 'ids'):
                    toks_up_to = toks_up_to[0].ids
                cum_lens.append(len(list(toks_up_to)))
            except Exception:
                cum_lens.append(cum_lens[-1])

        # For each assistant turn, also need the generation-prompt offset
        # (tokens added between user message and assistant response).
        # Compute once for the first assistant turn, reuse for others.
        gen_prompt_offset = 0
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                try:
                    prefix_with_gen = tokenizer.apply_chat_template(
                        messages[:i],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    if hasattr(prefix_with_gen, 'input_ids'):
                        prefix_with_gen = prefix_with_gen.input_ids
                    elif hasattr(prefix_with_gen, '__getitem__') and hasattr(prefix_with_gen[0], 'ids'):
                        prefix_with_gen = prefix_with_gen[0].ids
                    gen_prompt_offset = len(list(prefix_with_gen)) - cum_lens[i]
                except Exception:
                    gen_prompt_offset = 0
                break

        # Mark assistant tokens as trainable using pre-computed boundaries
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            # Assistant start = tokens up to messages[:i] + generation prompt
            start = cum_lens[i] + gen_prompt_offset
            # Assistant end = tokens up to messages[:i+1]
            end = cum_lens[i + 1]
            for j in range(start, min(end, len(input_ids))):
                labels[j] = input_ids[j]

    # Truncate
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    return {"input_ids": input_ids, "labels": labels}


def _tokenize_completions(
    sample: dict,
    tokenizer,
    mask_prompt: bool,
    max_seq_length: int,
) -> dict:
    """Tokenize a completions format sample."""
    messages = [
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["completion"]},
    ]
    chat_sample = {"messages": messages}
    return _tokenize_chat(chat_sample, tokenizer, mask_prompt, max_seq_length)


def _tokenize_text(
    sample: dict,
    tokenizer,
    max_seq_length: int,
) -> dict:
    """Tokenize a text format sample.

    Text format: train on all tokens (labels = input_ids).
    """
    text = sample["text"]
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Append EOS if not present
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if not tokens or tokens[-1] != tokenizer.eos_token_id:
            tokens.append(tokenizer.eos_token_id)

    # Truncate
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]

    # Text format: train on everything
    return {"input_ids": tokens, "labels": list(tokens)}


def _tokenize_preference(
    sample: dict,
    tokenizer,
    mask_prompt: bool,
    max_seq_length: int,
) -> dict:
    """Tokenize a preference format sample (DPO).

    Returns a dict with chosen and rejected tokenizations using per-token labels.
    """
    chosen = _tokenize_chat(
        {"messages": sample["chosen"]}, tokenizer, mask_prompt, max_seq_length)
    rejected = _tokenize_chat(
        {"messages": sample["rejected"]}, tokenizer, mask_prompt, max_seq_length)

    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_labels": chosen["labels"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_labels": rejected["labels"],
    }
