"""Vision model inference — VLM generation from image+text input.

Wraps mlx-vlm for loading and generating with vision-language models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx


def load_vision_model(model_path: str, adapter_path: Optional[str] = None):
    """Load a vision-language model via mlx-vlm.

    Args:
        model_path: HuggingFace model ID or local path
        adapter_path: Optional path to LoRA adapter weights

    Returns:
        (model, processor) tuple
    """
    from mlx_forge.vision import _check_mlx_vlm
    _check_mlx_vlm()

    from mlx_vlm import load

    model, processor = load(model_path)

    if adapter_path:
        adapter_file = Path(adapter_path) / "adapters.safetensors"
        if adapter_file.exists():
            weights = mx.load(str(adapter_file))
            model.load_weights(list(weights.items()), strict=False)

    model.eval()
    mx.eval(model.parameters())
    return model, processor


def generate_vision(
    model,
    processor,
    prompt: str,
    images: list,
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
):
    """Generate text from image+text input.

    Args:
        model: Loaded VLM model
        processor: VLM processor
        prompt: Text prompt
        images: List of image inputs (PIL Image, path, or URL)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text string
    """
    from mlx_forge.vision import _check_mlx_vlm
    _check_mlx_vlm()

    from mlx_vlm import generate

    return generate(
        model,
        processor,
        prompt,
        images,
        max_tokens=max_tokens,
        temp=temperature,
    )
