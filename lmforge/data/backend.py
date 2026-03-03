"""Arrow-based memory-mapped storage backend using the datasets library.

Replaces the old safetensors cache with HuggingFace datasets for
zero-copy memory-mapped access to tokenized data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DATASETS_DIR = "~/.lmforge/datasets"


def get_processed_path(dataset_name: str, model_id: str) -> Path:
    """Path for tokenized dataset: ~/.lmforge/datasets/processed/{name}--{model_slug}"""
    slug = model_id.replace("/", "--")
    return Path(DATASETS_DIR).expanduser() / "processed" / f"{dataset_name}--{slug}"


def save_tokenized(dataset_name: str, model_id: str, samples: list[dict]) -> Path:
    """Save tokenized samples as Arrow dataset.

    samples: list of {"input_ids": list[int], "labels": list[int]}
             or list of {"chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"}

    Returns the path where the dataset was saved.
    """
    from datasets import Dataset

    path = get_processed_path(dataset_name, model_id)

    # Detect columns from first sample
    if not samples:
        raise ValueError("Cannot save empty dataset")

    first = samples[0]
    columns = {}
    for key in first:
        columns[key] = [s[key] for s in samples]

    ds = Dataset.from_dict(columns)
    ds.save_to_disk(str(path))

    # Save metadata alongside
    _save_meta(path, dataset_name, model_id, samples)

    return path


def load_tokenized(dataset_name: str, model_id: str):
    """Load tokenized dataset (memory-mapped, ~0 RAM).

    Returns a datasets.Dataset object that can be iterated or indexed.
    """
    from datasets import load_from_disk

    path = get_processed_path(dataset_name, model_id)
    return load_from_disk(str(path))


def tokenized_exists(dataset_name: str, model_id: str) -> bool:
    """Check if a tokenized dataset exists."""
    path = get_processed_path(dataset_name, model_id)
    return path.exists() and (path / "dataset_info.json").exists()


def compute_fingerprint(data_path: str, tokenizer) -> str:
    """Compute cache fingerprint from data file, tokenizer vocab, and chat template.

    Returns a hex string: sha256(data_hash + tokenizer_hash + template_hash).
    """
    data_hash = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()

    vocab_items = sorted(tokenizer.get_vocab().items())
    vocab_str = json.dumps(vocab_items)
    tokenizer_hash = hashlib.sha256(vocab_str.encode()).hexdigest()

    template = getattr(tokenizer, "chat_template", None) or ""
    template_hash = hashlib.sha256(template.encode()).hexdigest()

    combined = data_hash + tokenizer_hash + template_hash
    fingerprint = hashlib.sha256(combined.encode()).hexdigest()

    return f"sha256:{fingerprint}"


def list_processed() -> list[dict]:
    """List all processed datasets with metadata."""
    processed_dir = Path(DATASETS_DIR).expanduser() / "processed"
    if not processed_dir.exists():
        return []

    results = []
    for entry in sorted(processed_dir.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta["path"] = str(entry)
            results.append(meta)
        except Exception:
            continue
    return results


def delete_processed(dataset_name: str, model_id: str) -> bool:
    """Delete a processed dataset. Returns True if deleted."""
    import shutil
    path = get_processed_path(dataset_name, model_id)
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def _save_meta(path: Path, dataset_name: str, model_id: str, samples: list[dict]):
    """Save metadata for a processed dataset."""
    first = samples[0]
    is_preference = "chosen_input_ids" in first

    if is_preference:
        all_lengths = [
            max(len(s["chosen_input_ids"]), len(s["rejected_input_ids"]))
            for s in samples
        ]
        total_tokens = sum(
            len(s["chosen_input_ids"]) + len(s["rejected_input_ids"])
            for s in samples
        )
    else:
        all_lengths = [len(s["input_ids"]) for s in samples]
        total_tokens = sum(all_lengths)

    meta = {
        "schema_version": 2,
        "dataset_name": dataset_name,
        "model_id": model_id,
        "num_samples": len(samples),
        "total_tokens": total_tokens,
        "max_length": max(all_lengths) if all_lengths else 0,
        "min_length": min(all_lengths) if all_lengths else 0,
        "mean_length": total_tokens / len(samples) if samples else 0.0,
        "format": "preference" if is_preference else "sft",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    meta_path = path / "meta.json"
    path.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
