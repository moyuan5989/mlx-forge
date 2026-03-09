#!/usr/bin/env python3
"""Download and convert HuggingFace datasets to LMForge JSONL format.

This is a standalone script (not part of lmforge package). Install dependencies:
    pip install datasets tqdm

Usage:
    python scripts/download_hf_dataset.py alpaca --output data/
    python scripts/download_hf_dataset.py openassistant --output data/
    python scripts/download_hf_dataset.py custom --dataset user/repo --output data/

Supported presets:
    alpaca          - Stanford Alpaca instruction dataset (52K samples)
    openassistant   - OpenAssistant conversations (subset)
    dolly           - Databricks Dolly 15K instruction dataset
    custom          - Any HF dataset (requires --dataset and format mapping)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install datasets tqdm")
    sys.exit(1)


# ============================================================================
# Dataset Presets
# ============================================================================

def convert_alpaca(example: dict) -> dict:
    """Convert Alpaca format to LMForge chat format.

    Alpaca schema: {instruction, input (optional), output}
    """
    instruction = example["instruction"]
    user_input = example.get("input", "").strip()

    # Combine instruction and input if present
    if user_input:
        user_message = f"{instruction}\n\n{user_input}"
    else:
        user_message = instruction

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def convert_openassistant(example: dict) -> dict | None:
    """Convert OpenAssistant format to LMForge chat format.

    OpenAssistant has multi-turn conversations. We extract user-assistant pairs.
    """
    messages = []

    # OpenAssistant stores conversations as lists
    if "messages" in example:
        for msg in example["messages"]:
            role = msg.get("role")
            content = msg.get("content", "").strip()

            if not content:
                continue

            # Map roles
            if role == "prompter":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})

    # Only return if we have at least one exchange
    if len(messages) >= 2:
        return {"messages": messages}

    return None


def convert_dolly(example: dict) -> dict:
    """Convert Databricks Dolly format to LMForge chat format.

    Dolly schema: {instruction, context (optional), response}
    """
    instruction = example["instruction"]
    context = example.get("context", "").strip()

    # Combine instruction and context if present
    if context:
        user_message = f"{instruction}\n\nContext:\n{context}"
    else:
        user_message = instruction

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": example["response"]},
        ]
    }


def convert_sharegpt(example: dict) -> dict | None:
    """Convert ShareGPT format to LMForge chat format.

    ShareGPT schema: {conversations: [{from: human/gpt, value: text}]}
    """
    if "conversations" not in example:
        return None

    messages = []
    for msg in example["conversations"]:
        role = msg.get("from")
        content = msg.get("value", "").strip()

        if not content:
            continue

        # Map roles
        if role in ("human", "user"):
            messages.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": content})

    if len(messages) >= 2:
        return {"messages": messages}

    return None


# Preset configurations
PRESETS = {
    "alpaca": {
        "dataset": "tatsu-lab/alpaca",
        "converter": convert_alpaca,
        "train_split": "train",
        "val_split": None,  # No validation split
        "val_ratio": 0.05,  # Create 5% validation split
    },
    "openassistant": {
        "dataset": "OpenAssistant/oasst1",
        "converter": convert_openassistant,
        "train_split": "train",
        "val_split": "validation",
        "val_ratio": None,
    },
    "dolly": {
        "dataset": "databricks/databricks-dolly-15k",
        "converter": convert_dolly,
        "train_split": "train",
        "val_split": None,
        "val_ratio": 0.05,
    },
}


# ============================================================================
# Main Conversion Logic
# ============================================================================

def convert_dataset(
    dataset_name: str,
    converter: Callable,
    train_split: str,
    val_split: str | None,
    val_ratio: float | None,
    output_dir: Path,
    max_samples: int | None = None,
    subset: str | None = None,
):
    """Download and convert a HuggingFace dataset to JSONL."""

    print(f"Loading dataset: {dataset_name}")
    if subset:
        print(f"  Subset: {subset}")

    # Load dataset
    try:
        if subset:
            ds = load_dataset(dataset_name, subset)
        else:
            ds = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTry specifying a subset with --subset if the dataset requires one.")
        sys.exit(1)

    print(f"Available splits: {list(ds.keys())}")

    # Process training split
    train_samples = []
    if train_split in ds:
        print(f"\nProcessing {train_split} split...")
        train_data = ds[train_split]

        if max_samples:
            train_data = train_data.select(range(min(max_samples, len(train_data))))

        for example in tqdm(train_data, desc="Converting"):
            converted = converter(example)
            if converted:
                train_samples.append(converted)

        print(f"  Converted: {len(train_samples)} samples")

    # Process or create validation split
    val_samples = []

    if val_split and val_split in ds:
        # Use existing validation split
        print(f"\nProcessing {val_split} split...")
        val_data = ds[val_split]

        if max_samples:
            val_data = val_data.select(range(min(max_samples // 10, len(val_data))))

        for example in tqdm(val_data, desc="Converting"):
            converted = converter(example)
            if converted:
                val_samples.append(converted)

        print(f"  Converted: {len(val_samples)} samples")

    elif val_ratio and train_samples:
        # Create validation split from training data
        print(f"\nCreating validation split ({val_ratio*100:.0f}% of training data)...")
        split_idx = int(len(train_samples) * (1 - val_ratio))
        val_samples = train_samples[split_idx:]
        train_samples = train_samples[:split_idx]
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Validation: {len(val_samples)} samples")

    # Write JSONL files
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    print(f"\nWriting {train_path}")
    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if val_samples:
        val_path = output_dir / "valid.jsonl"
        print(f"Writing {val_path}")
        with open(val_path, "w") as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("\n✓ Dataset conversion complete!")
    print("\nNext steps:")
    print(f"  lmforge prepare --data {train_path} --model <your-model>")
    print("  lmforge train --config train.yaml")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and convert HuggingFace datasets to LMForge JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "preset",
        choices=list(PRESETS.keys()) + ["custom"],
        help="Dataset preset or 'custom' for custom dataset",
    )

    parser.add_argument(
        "--dataset",
        help="HuggingFace dataset name (required for 'custom' preset)",
    )

    parser.add_argument(
        "--subset",
        help="Dataset subset/configuration name (e.g., 'english' for some datasets)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output directory for JSONL files (default: data/)",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to convert (for testing)",
    )

    parser.add_argument(
        "--format",
        choices=["alpaca", "openassistant", "dolly", "sharegpt"],
        help="Format converter to use for custom dataset",
    )

    args = parser.parse_args()

    # Get preset or build custom config
    if args.preset == "custom":
        if not args.dataset:
            print("Error: --dataset required for custom preset")
            sys.exit(1)

        if not args.format:
            print("Error: --format required for custom preset")
            print("Available formats: alpaca, openassistant, dolly, sharegpt")
            sys.exit(1)

        # Map format to converter
        format_converters = {
            "alpaca": convert_alpaca,
            "openassistant": convert_openassistant,
            "dolly": convert_dolly,
            "sharegpt": convert_sharegpt,
        }

        config = {
            "dataset": args.dataset,
            "converter": format_converters[args.format],
            "train_split": "train",
            "val_split": "validation",
            "val_ratio": 0.05,
        }
    else:
        config = PRESETS[args.preset]

    # Convert dataset
    convert_dataset(
        dataset_name=config["dataset"],
        converter=config["converter"],
        train_split=config["train_split"],
        val_split=config["val_split"],
        val_ratio=config["val_ratio"],
        output_dir=args.output,
        max_samples=args.max_samples,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
