"""Prompt cache — save/load KV cache state to disk.

Useful for reusing prefill results across generations:
- System prompts that don't change between requests
- Few-shot examples
- Long context reuse
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
from safetensors.mlx import load_file, save_file


def save_prompt_cache(
    path: str | Path,
    cache: list,
    metadata: Optional[dict] = None,
) -> Path:
    """Save KV cache state to safetensors file with metadata.

    Args:
        path: Output path for the cache file
        cache: List of KVCache instances (one per layer)
        metadata: Optional metadata dict (model name, offset, etc.)

    Returns:
        Path to the saved cache file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect cache arrays
    tensors = {}
    offsets = []

    for i, c in enumerate(cache):
        if c.keys is not None:
            # Only save the filled portion of the cache
            tensors[f"layer.{i}.keys"] = c.keys[:, :, :c.offset, :]
            tensors[f"layer.{i}.values"] = c.values[:, :, :c.offset, :]
        offsets.append(c.offset)

    # Save tensors
    save_file(tensors, str(path))

    # Save metadata alongside
    meta = metadata or {}
    meta["offsets"] = offsets
    meta["num_layers"] = len(cache)

    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return path


def load_prompt_cache(path: str | Path) -> tuple[dict, dict]:
    """Load prompt cache from file.

    Args:
        path: Path to the cache safetensors file

    Returns:
        (tensors, metadata): tensors dict and metadata dict
    """
    path = Path(path)

    tensors = load_file(str(path))

    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return tensors, metadata


def apply_prompt_cache(cache: list, tensors: dict, metadata: dict) -> None:
    """Apply loaded prompt cache tensors to KVCache instances.

    Args:
        cache: List of KVCache instances to populate
        tensors: Loaded tensors from load_prompt_cache()
        metadata: Metadata dict with offsets
    """
    offsets = metadata.get("offsets", [])

    for i, c in enumerate(cache):
        key_name = f"layer.{i}.keys"
        val_name = f"layer.{i}.values"

        if key_name in tensors and val_name in tensors:
            keys = tensors[key_name]
            values = tensors[val_name]

            if c._max_size > 0:
                # Pre-allocated path
                L = keys.shape[2]
                if not c._allocated:
                    B, H, _, D = keys.shape
                    c.keys = mx.zeros((B, H, c._max_size, D), dtype=keys.dtype)
                    c.values = mx.zeros((B, H, c._max_size, D), dtype=values.dtype)
                    c._allocated = True
                c.keys[:, :, :L, :] = keys
                c.values[:, :, :L, :] = values
                c.offset = L
            else:
                # Concatenation path
                c.keys = keys
                c.values = values
                c.offset = keys.shape[2]
        elif i < len(offsets):
            c.offset = offsets[i]
