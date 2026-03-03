"""Dataset format detection and validation for LMForge v0."""

from __future__ import annotations

from typing import Literal


def detect_format(samples: list[dict]) -> Literal["chat", "completions", "text", "preference"]:
    """Auto-detect dataset format from the first sample's keys.

    - Has "chosen" and "rejected" -> preference format (DPO)
    - Has "messages" -> chat format
    - Has "prompt" and "completion" -> completions format
    - Has "text" -> text format
    - Otherwise -> raise error listing found keys
    """
    if not samples:
        raise ValueError("Cannot detect format from empty dataset.")

    first_sample = samples[0]
    keys = set(first_sample.keys())

    if "chosen" in keys and "rejected" in keys:
        return "preference"
    elif "messages" in keys:
        return "chat"
    elif "prompt" in keys and "completion" in keys:
        return "completions"
    elif "text" in keys:
        return "text"
    else:
        raise ValueError(
            f"Unknown dataset format. Expected 'messages', 'prompt'+'completion', "
            f"'text', or 'chosen'+'rejected' keys. Found keys: {sorted(keys)}"
        )


def validate_samples(samples: list[dict], fmt: str) -> list[str]:
    """Validate all samples match the detected format schema.

    Returns a list of error messages (empty if all valid).
    Iterates all samples and collects all errors before reporting.
    """
    errors = []

    for idx, sample in enumerate(samples):
        if fmt == "chat":
            errors.extend(_validate_chat_sample(sample, idx))
        elif fmt == "completions":
            errors.extend(_validate_completions_sample(sample, idx))
        elif fmt == "text":
            errors.extend(_validate_text_sample(sample, idx))
        elif fmt == "preference":
            errors.extend(_validate_preference_sample(sample, idx))
        else:
            errors.append(f"Unknown format: {fmt}")
            break

    return errors


def _validate_chat_sample(sample: dict, idx: int) -> list[str]:
    """Validate a single chat format sample."""
    errors = []

    if "messages" not in sample:
        errors.append(f"Sample {idx}: missing 'messages' field")
        return errors

    messages = sample["messages"]
    if not isinstance(messages, list):
        errors.append(f"Sample {idx}: 'messages' must be a list, got {type(messages).__name__}")
        return errors

    if not messages:
        errors.append(f"Sample {idx}: 'messages' list is empty")
        return errors

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(
                f"Sample {idx}, message {msg_idx}: message must be a dict, "
                f"got {type(msg).__name__}"
            )
            continue

        if "role" not in msg:
            errors.append(f"Sample {idx}, message {msg_idx}: missing 'role' field")
        elif not isinstance(msg["role"], str):
            errors.append(
                f"Sample {idx}, message {msg_idx}: 'role' must be a string, "
                f"got {type(msg['role']).__name__}"
            )

        if "content" not in msg:
            errors.append(f"Sample {idx}, message {msg_idx}: missing 'content' field")
        elif not isinstance(msg["content"], str):
            errors.append(
                f"Sample {idx}, message {msg_idx}: 'content' must be a string, "
                f"got {type(msg['content']).__name__}"
            )

    return errors


def _validate_completions_sample(sample: dict, idx: int) -> list[str]:
    """Validate a single completions format sample."""
    errors = []

    if "prompt" not in sample:
        errors.append(f"Sample {idx}: missing 'prompt' field")
    elif not isinstance(sample["prompt"], str):
        errors.append(
            f"Sample {idx}: 'prompt' must be a string, got {type(sample['prompt']).__name__}"
        )

    if "completion" not in sample:
        errors.append(f"Sample {idx}: missing 'completion' field")
    elif not isinstance(sample["completion"], str):
        errors.append(
            f"Sample {idx}: 'completion' must be a string, "
            f"got {type(sample['completion']).__name__}"
        )

    return errors


def _validate_text_sample(sample: dict, idx: int) -> list[str]:
    """Validate a single text format sample."""
    errors = []

    if "text" not in sample:
        errors.append(f"Sample {idx}: missing 'text' field")
    elif not isinstance(sample["text"], str):
        errors.append(
            f"Sample {idx}: 'text' must be a string, got {type(sample['text']).__name__}"
        )

    return errors


def _validate_preference_sample(sample: dict, idx: int) -> list[str]:
    """Validate a single preference format sample (DPO).

    Expected format:
    {"chosen": [{"role": "user", ...}, {"role": "assistant", ...}],
     "rejected": [{"role": "user", ...}, {"role": "assistant", ...}]}
    """
    errors = []

    for field in ("chosen", "rejected"):
        if field not in sample:
            errors.append(f"Sample {idx}: missing '{field}' field")
            continue

        messages = sample[field]
        if not isinstance(messages, list):
            errors.append(
                f"Sample {idx}: '{field}' must be a list, got {type(messages).__name__}"
            )
            continue

        if not messages:
            errors.append(f"Sample {idx}: '{field}' list is empty")
            continue

        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(
                    f"Sample {idx}, {field}[{msg_idx}]: message must be a dict, "
                    f"got {type(msg).__name__}"
                )
                continue

            if "role" not in msg:
                errors.append(f"Sample {idx}, {field}[{msg_idx}]: missing 'role' field")
            elif not isinstance(msg["role"], str):
                errors.append(
                    f"Sample {idx}, {field}[{msg_idx}]: 'role' must be a string, "
                    f"got {type(msg['role']).__name__}"
                )

            if "content" not in msg:
                errors.append(f"Sample {idx}, {field}[{msg_idx}]: missing 'content' field")
            elif not isinstance(msg["content"], str):
                errors.append(
                    f"Sample {idx}, {field}[{msg_idx}]: 'content' must be a string, "
                    f"got {type(msg['content']).__name__}"
                )

    return errors
