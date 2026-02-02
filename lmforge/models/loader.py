"""Model and tokenizer loading for LMForge v0."""

from __future__ import annotations

from transformers import AutoTokenizer


def load_model(model_path: str, *, trust_remote_code: bool = False):
    """Load a model and tokenizer from a HuggingFace repo ID or local path.

    Returns (model, tokenizer) tuple.

    Uses mlx_lm for model loading. If mlx_lm is not installed, raises ImportError
    with installation instructions.
    """
    try:
        from mlx_lm import load as mlx_lm_load
    except ImportError:
        raise ImportError(
            "mlx_lm is required for model loading. Install it with:\n"
            "  pip install mlx-lm\n\n"
            "Note: mlx_lm is not a hard dependency of lmforge to keep the core "
            "package lightweight. It's only needed when loading models for training."
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    # Load model using mlx_lm
    # mlx_lm.load returns (model, tokenizer), but we already loaded the tokenizer
    # to have consistent behavior. Use mlx_lm's model loading.
    model, _ = mlx_lm_load(
        model_path,
        tokenizer_config={"trust_remote_code": trust_remote_code},
    )

    return model, tokenizer
