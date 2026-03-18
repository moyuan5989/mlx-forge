"""Adapter fuse and export — merge LoRA weights into base model.

Walks the model tree, calls .fuse() on LoRALinear/LoRAEmbedding modules,
and saves the resulting plain model as safetensors.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_forge.adapters.lora import LoRAEmbedding, LoRALinear


def fuse_model(model: nn.Module) -> nn.Module:
    """Walk all modules, call .fuse() on LoRA wrappers, return plain model.

    Args:
        model: Model with LoRA adapters applied.

    Returns:
        The same model with all LoRA modules replaced by fused Linear/Embedding.
    """
    fused_count = 0
    visited: set[int] = set()

    def _fuse_recursive(module: nn.Module) -> None:
        nonlocal fused_count
        if id(module) in visited:
            return
        visited.add(id(module))

        # Use MLX's named_modules() to get direct children only
        try:
            children = [(n, m) for n, m in module.named_modules() if "." not in n]
        except Exception:
            children = []

        for child_name, child in children:
            if isinstance(child, (LoRALinear, LoRAEmbedding)):
                fused_count += 1
                setattr(module, child_name, child.fuse())
            else:
                _fuse_recursive(child)

    _fuse_recursive(model)
    print(f"Fused {fused_count} LoRA modules")
    return model


def save_fused_model(
    model: nn.Module,
    tokenizer_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Save fused model weights and tokenizer files.

    Args:
        model: Fused model (after calling fuse_model).
        tokenizer_path: Directory containing tokenizer files (from HF cache or local).
        output_dir: Where to save the exported model.

    Returns:
        Path to the output directory.
    """
    from safetensors.mlx import save_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights = dict(tree_flatten(model.parameters()))
    save_file(weights, str(output_dir / "model.safetensors"))

    # Copy tokenizer files
    tokenizer_path = Path(tokenizer_path)
    tokenizer_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",  # for sentencepiece-based models
    ]
    for fname in tokenizer_files:
        src = tokenizer_path / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    print(f"Saved fused model to {output_dir}")
    return output_dir
