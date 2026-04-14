"""Fill-in-middle (FIM) support for code completion models.

Builds FIM prompts using model-specific templates. Supports CodeLlama,
DeepSeek, StarCoder, and Qwen code models.
"""

from __future__ import annotations

# FIM template patterns for different model families
FIM_TEMPLATES: dict[str, dict[str, str]] = {
    "codellama": {
        "prefix": "<PRE>",
        "suffix": "<SUF>",
        "middle": "<MID>",
    },
    "deepseek": {
        "prefix": "<｜fim▁begin｜>",
        "suffix": "<｜fim▁hole｜>",
        "middle": "<｜fim▁end｜>",
    },
    "starcoder": {
        "prefix": "<fim_prefix>",
        "suffix": "<fim_suffix>",
        "middle": "<fim_middle>",
    },
    "qwen": {
        "prefix": "<|fim_prefix|>",
        "suffix": "<|fim_suffix|>",
        "middle": "<|fim_middle|>",
    },
}

# Map model_type / model_id patterns to FIM template names
_MODEL_FIM_MAP: list[tuple[str, str]] = [
    ("codellama", "codellama"),
    ("code_llama", "codellama"),
    ("deepseek", "deepseek"),
    ("starcoder", "starcoder"),
    ("qwen", "qwen"),
]


def detect_fim_support(config: dict) -> str | None:
    """Check if a model supports FIM and return template name.

    Args:
        config: Model config dict (from config.json).

    Returns:
        Template name (e.g., "codellama") or None if FIM not supported.
    """
    model_type = config.get("model_type", "").lower()
    model_id = config.get("_name_or_path", "").lower()

    for pattern, template_name in _MODEL_FIM_MAP:
        if pattern in model_type or pattern in model_id:
            return template_name

    return None


def build_fim_prompt(prefix: str, suffix: str, template_name: str) -> str:
    """Build a fill-in-middle prompt for the given model family.

    Args:
        prefix: Code before the cursor.
        suffix: Code after the cursor.
        template_name: FIM template name from detect_fim_support().

    Returns:
        Formatted FIM prompt string.

    Raises:
        ValueError: If template_name is not recognized.
    """
    template = FIM_TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(
            f"Unknown FIM template '{template_name}'. "
            f"Available: {sorted(FIM_TEMPLATES.keys())}"
        )

    return f"{template['prefix']}{prefix}{template['suffix']}{suffix}{template['middle']}"
