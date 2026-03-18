"""HuggingFace Datasets loader for MLX Forge.

Downloads and converts HF datasets to MLX Forge JSONL format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_hf_dataset(
    dataset_id: str,
    *,
    split: str = "train",
    subset: Optional[str] = None,
    columns: Optional[dict[str, str]] = None,
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
) -> list[dict]:
    """Load a HuggingFace dataset and convert to MLX Forge format.

    Args:
        dataset_id: HF dataset ID (e.g., "tatsu-lab/alpaca")
        split: Dataset split to load
        subset: Dataset subset/config name
        columns: Column mapping override (e.g., {"instruction": "input", "output": "target"})
        max_samples: Limit number of samples
        token: HF API token for gated datasets

    Returns:
        List of dicts in MLX Forge format (messages or text format).
    """
    from datasets import load_dataset

    kwargs = {"split": split}
    if subset:
        kwargs["name"] = subset
    if token:
        kwargs["token"] = token

    ds = load_dataset(dataset_id, **kwargs)

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    # Auto-detect format if no column mapping provided
    if columns is None:
        mapping = auto_detect_columns(ds)
    else:
        mapping = columns

    return apply_column_mapping(ds, mapping)


def auto_detect_columns(dataset) -> dict[str, str]:
    """Auto-detect column format from dataset columns.

    Detects:
    - OpenAI format: "messages" column
    - Alpaca format: "instruction" + "output" columns
    - ShareGPT format: "conversations" column
    - Text format: "text" column
    - Preference format: "prompt" + "chosen" + "rejected" columns

    Returns:
        Column mapping dict with detected format info.
    """
    cols = set(dataset.column_names)

    # OpenAI messages format
    if "messages" in cols:
        return {"format": "messages", "messages": "messages"}

    # ShareGPT format
    if "conversations" in cols:
        return {"format": "sharegpt", "conversations": "conversations"}

    # Preference format (DPO)
    if "prompt" in cols and "chosen" in cols and "rejected" in cols:
        return {"format": "preference", "prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}

    # Alpaca format
    if "instruction" in cols and "output" in cols:
        mapping = {"format": "alpaca", "instruction": "instruction", "output": "output"}
        if "input" in cols:
            mapping["input"] = "input"
        return mapping

    # Text format
    if "text" in cols:
        return {"format": "text", "text": "text"}

    # Fallback: try common variations
    if "question" in cols and "answer" in cols:
        return {"format": "qa", "question": "question", "answer": "answer"}

    if "content" in cols:
        return {"format": "text", "text": "content"}

    raise ValueError(
        f"Cannot auto-detect dataset format. Columns: {sorted(cols)}. "
        f"Use --hf-columns to specify mapping."
    )


def apply_column_mapping(dataset, mapping: dict) -> list[dict]:
    """Convert dataset rows using column mapping to MLX Forge format.

    Returns list of dicts in chat format (messages) or text format.
    """
    fmt = mapping.get("format", "text")
    samples = []

    for row in dataset:
        if fmt == "messages":
            # Already in OpenAI format
            messages = row[mapping["messages"]]
            if isinstance(messages, list) and len(messages) > 0:
                samples.append({"messages": messages})

        elif fmt == "sharegpt":
            # Convert ShareGPT to OpenAI format
            convos = row[mapping["conversations"]]
            if isinstance(convos, list):
                messages = []
                for turn in convos:
                    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
                    role = turn.get("from", turn.get("role", "user"))
                    role = role_map.get(role, role)
                    content = turn.get("value", turn.get("content", ""))
                    messages.append({"role": role, "content": content})
                if messages:
                    samples.append({"messages": messages})

        elif fmt == "alpaca":
            # Convert Alpaca to chat format
            instruction = row[mapping["instruction"]]
            output = row[mapping["output"]]
            input_text = row.get(mapping.get("input", "input"), "")

            user_content = instruction
            if input_text:
                user_content = f"{instruction}\n\n{input_text}"

            samples.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output},
                ]
            })

        elif fmt == "preference":
            # DPO/preference format
            samples.append({
                "prompt": row[mapping["prompt"]],
                "chosen": row[mapping["chosen"]],
                "rejected": row[mapping["rejected"]],
            })

        elif fmt == "qa":
            # Q&A to chat format
            samples.append({
                "messages": [
                    {"role": "user", "content": row[mapping["question"]]},
                    {"role": "assistant", "content": row[mapping["answer"]]},
                ]
            })

        elif fmt == "text":
            # Raw text
            samples.append({"text": row[mapping["text"]]})

    return samples


def save_as_jsonl(samples: list[dict], output_path: str | Path) -> Path:
    """Save samples as JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return output_path
