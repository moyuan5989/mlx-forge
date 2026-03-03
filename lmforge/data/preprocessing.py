"""Tokenization with per-token labels for LMForge V2.

Returns {"input_ids": list[int], "labels": list[int]} where labels use
-100 to mask tokens that should not contribute to loss. This enables
training on ALL assistant turns in multi-turn conversations.
"""

from __future__ import annotations


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

        # Build up the conversation incrementally to find boundaries
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # Tokenize messages[:i] with generation prompt to find where assistant starts
            try:
                prefix_tokens = tokenizer.apply_chat_template(
                    messages[:i],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if hasattr(prefix_tokens, 'input_ids'):
                    prefix_tokens = prefix_tokens.input_ids
                elif hasattr(prefix_tokens, '__getitem__') and hasattr(prefix_tokens[0], 'ids'):
                    prefix_tokens = prefix_tokens[0].ids
                prefix_len = len(list(prefix_tokens))
            except Exception:
                prefix_len = 0

            # Tokenize messages[:i+1] to find where assistant ends
            try:
                full_up_to = tokenizer.apply_chat_template(
                    messages[:i + 1],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                if hasattr(full_up_to, 'input_ids'):
                    full_up_to = full_up_to.input_ids
                elif hasattr(full_up_to, '__getitem__') and hasattr(full_up_to[0], 'ids'):
                    full_up_to = full_up_to[0].ids
                end_len = len(list(full_up_to))
            except Exception:
                end_len = len(input_ids)

            # Mark assistant tokens as trainable
            for j in range(prefix_len, min(end_len, len(input_ids))):
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
