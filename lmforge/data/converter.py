"""Dataset format converters for LMForge.

Converts HuggingFace dataset columns to LMForge standard format
(chat, completions, text, preference) before tokenization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmforge.data.catalog import DatasetProfile


def convert_dataset(hf_dataset, profile: "DatasetProfile") -> list[dict]:
    """Convert HF dataset to LMForge standard format (raw, pre-tokenization).

    Returns list of dicts in chat/completions/text/preference format.
    """
    if profile.columns is None:
        raise ValueError(f"Dataset '{profile.id}' has no column mapping defined")

    converter = CONVERTERS.get(profile.columns.type)
    if converter is None:
        raise ValueError(
            f"Unknown converter type '{profile.columns.type}'. "
            f"Available: {', '.join(CONVERTERS.keys())}"
        )

    return converter(hf_dataset, profile.columns.mapping)


def _convert_rename(hf_dataset, mapping: dict) -> list[dict]:
    """Simple column rename: map source columns to LMForge standard names.

    Example mapping: {"instruction": "prompt", "response": "completion"}
    """
    results = []
    for row in hf_dataset:
        sample = {}
        for src_col, dst_col in mapping.items():
            if src_col in row:
                sample[dst_col] = row[src_col]
        results.append(sample)
    return results


def _convert_alpaca(hf_dataset, mapping: dict) -> list[dict]:
    """Convert Alpaca-style format: instruction + optional input -> prompt.

    If input is non-empty, appends it to the instruction.
    """
    inst_col = mapping.get("instruction", "instruction")
    input_col = mapping.get("input", "input")
    output_col = mapping.get("output", "output")

    results = []
    for row in hf_dataset:
        instruction = row.get(inst_col, "")
        inp = row.get(input_col, "")
        output = row.get(output_col, "")

        if inp and inp.strip():
            prompt = f"{instruction}\n\n{inp}"
        else:
            prompt = instruction

        results.append({"prompt": prompt, "completion": output})
    return results


def _convert_chat_messages(hf_dataset, mapping: dict) -> list[dict]:
    """Rename the messages column to 'messages'.

    Handles datasets where messages are already in role/content format.
    """
    msg_col = mapping.get("messages", "messages")

    results = []
    for row in hf_dataset:
        messages = row.get(msg_col, [])
        # Ensure messages are dicts with role/content
        normalized = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                normalized.append({"role": msg["role"], "content": msg["content"]})
        if normalized:
            results.append({"messages": normalized})
    return results


def _convert_sharegpt(hf_dataset, mapping: dict) -> list[dict]:
    """Convert ShareGPT format: conversations with from/value -> role/content.

    ShareGPT uses "from" (human/gpt/system) and "value" fields.
    """
    conv_col = mapping.get("conversations", "conversations")

    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }

    results = []
    for row in hf_dataset:
        conversations = row.get(conv_col, [])
        messages = []
        for turn in conversations:
            if isinstance(turn, dict):
                role_key = turn.get("from", turn.get("role", ""))
                content = turn.get("value", turn.get("content", ""))
                role = role_map.get(role_key, role_key)
                if role and content:
                    messages.append({"role": role, "content": content})
        if messages:
            results.append({"messages": messages})
    return results


def _convert_text(hf_dataset, mapping: dict) -> list[dict]:
    """Convert a single text column to LMForge text format."""
    text_col = mapping.get("text", "text")

    results = []
    for row in hf_dataset:
        text = row.get(text_col, "")
        if text:
            results.append({"text": text})
    return results


def _convert_preference(hf_dataset, mapping: dict) -> list[dict]:
    """Convert preference datasets for DPO training.

    Handles common formats:
    - chosen/rejected as message lists
    - chosen/rejected as plain strings (wraps in messages)
    """
    chosen_col = mapping.get("chosen", "chosen")
    rejected_col = mapping.get("rejected", "rejected")

    results = []
    for row in hf_dataset:
        chosen = row.get(chosen_col)
        rejected = row.get(rejected_col)

        if chosen is None or rejected is None:
            continue

        # Handle message lists
        if isinstance(chosen, list) and isinstance(rejected, list):
            # Already message format
            chosen_msgs = _normalize_messages(chosen)
            rejected_msgs = _normalize_messages(rejected)
        elif isinstance(chosen, str) and isinstance(rejected, str):
            # Plain string: wrap in simple prompt/response
            prompt = row.get("prompt", row.get("question", row.get("input", "")))
            if isinstance(prompt, str) and prompt:
                chosen_msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ]
                rejected_msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected},
                ]
            else:
                chosen_msgs = [{"role": "assistant", "content": chosen}]
                rejected_msgs = [{"role": "assistant", "content": rejected}]
        else:
            continue

        if chosen_msgs and rejected_msgs:
            results.append({
                "chosen": chosen_msgs,
                "rejected": rejected_msgs,
            })

    return results


def _normalize_messages(messages: list) -> list[dict]:
    """Normalize a list of messages to role/content dicts."""
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("from", ""))
            content = msg.get("content", msg.get("value", ""))
            if role and content:
                result.append({"role": role, "content": content})
    return result


CONVERTERS = {
    "rename": _convert_rename,
    "alpaca": _convert_alpaca,
    "chat_messages": _convert_chat_messages,
    "sharegpt": _convert_sharegpt,
    "text_column": _convert_text,
    "preference": _convert_preference,
}
