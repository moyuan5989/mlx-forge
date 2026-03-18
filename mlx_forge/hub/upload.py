"""Upload models and adapters to HuggingFace Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def push_to_hub(
    local_dir: str | Path,
    repo_id: str,
    *,
    adapter_only: bool = False,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Push a model or adapter to HuggingFace Hub.

    Args:
        local_dir: Directory containing model files (model.safetensors, tokenizer, etc.)
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        adapter_only: If True, only upload adapter files (adapters.safetensors + config)
        private: Create private repository
        token: HuggingFace API token (uses cached token if None)

    Returns:
        URL of the uploaded repository.
    """
    from huggingface_hub import HfApi

    local_dir = Path(local_dir)
    api = HfApi(token=token)

    # Create or get repo
    repo_url = api.create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,
    )

    if adapter_only:
        # Upload only adapter files
        allow_patterns = [
            "adapters.safetensors",
            "adapter_config.json",
            "README.md",
        ]
    else:
        # Upload everything
        allow_patterns = None

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        allow_patterns=allow_patterns,
    )

    return repo_url.url if hasattr(repo_url, "url") else f"https://huggingface.co/{repo_id}"


def generate_model_card(
    config: dict,
    metrics: Optional[dict] = None,
    base_model: Optional[str] = None,
) -> str:
    """Generate a HuggingFace model card (README.md) from training config.

    Args:
        config: Training config dict
        metrics: Optional training metrics (final loss, etc.)
        base_model: Base model ID

    Returns:
        Rendered model card as string.
    """
    import jinja2

    template_str = _MODEL_CARD_TEMPLATE
    template = jinja2.Template(template_str)

    # Extract info from config
    adapter_config = config.get("adapter", {})
    training_config = config.get("training", {})
    model_config = config.get("model", {})

    base_model = base_model or model_config.get("path", "unknown")

    return template.render(
        base_model=base_model,
        adapter_method=adapter_config.get("method", "lora"),
        adapter_rank=adapter_config.get("rank", 8),
        adapter_scale=adapter_config.get("scale", 20.0),
        adapter_preset=adapter_config.get("preset"),
        adapter_targets=adapter_config.get("targets"),
        learning_rate=training_config.get("learning_rate", 1e-5),
        batch_size=training_config.get("batch_size", 2),
        num_iters=training_config.get("num_iters", 1000),
        optimizer=training_config.get("optimizer", "adam"),
        max_seq_length=config.get("data", {}).get("max_seq_length", 2048),
        final_train_loss=metrics.get("final_train_loss") if metrics else None,
        best_val_loss=metrics.get("best_val_loss") if metrics else None,
    )


_MODEL_CARD_TEMPLATE = """\
---
library_name: mlx-forge
tags:
  - mlx
  - lora
  - fine-tuned
  - apple-silicon
base_model: {{ base_model }}
---

# {{ base_model }} — Fine-tuned with MLX Forge

This model was fine-tuned using [MLX Forge](https://github.com/moyuan5989/mlx-forge) on Apple Silicon.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `{{ base_model }}` |
| Method | {{ adapter_method }} |
| Rank | {{ adapter_rank }} |
| Scale | {{ adapter_scale }} |
{% if adapter_preset %}| Preset | {{ adapter_preset }} |
{% endif %}{% if adapter_targets %}| Targets | {{ adapter_targets | join(', ') }} |
{% endif %}| Learning Rate | {{ learning_rate }} |
| Batch Size | {{ batch_size }} |
| Iterations | {{ num_iters }} |
| Optimizer | {{ optimizer }} |
| Max Seq Length | {{ max_seq_length }} |
{% if final_train_loss is not none %}| Final Train Loss | {{ "%.4f" | format(final_train_loss) }} |
{% endif %}{% if best_val_loss is not none %}| Best Val Loss | {{ "%.4f" | format(best_val_loss) }} |
{% endif %}

## Usage

```python
import mlx_forge

result = mlx_forge.generate(
    model="{{ base_model }}",
    adapter="path/to/adapters",
    prompt="Your prompt here",
)
print(result.text)
```

## Framework

- **Framework**: [MLX Forge](https://github.com/moyuan5989/mlx-forge)
- **Hardware**: Apple Silicon (M-series)
- **Backend**: [MLX](https://github.com/ml-explore/mlx)
"""


def push_adapter_only(
    adapter_dir: str | Path,
    repo_id: str,
    config: dict,
    *,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Upload just adapter files to HuggingFace Hub.

    Creates adapter_config.json and README.md, then uploads.

    Args:
        adapter_dir: Directory containing adapters.safetensors
        repo_id: HuggingFace repo ID
        config: Training config dict
        private: Create private repository
        token: HuggingFace API token

    Returns:
        URL of the uploaded repository.
    """
    import json

    adapter_dir = Path(adapter_dir)

    # Write adapter_config.json
    adapter_config = {
        "method": config.get("adapter", {}).get("method", "lora"),
        "rank": config.get("adapter", {}).get("rank", 8),
        "scale": config.get("adapter", {}).get("scale", 20.0),
        "base_model": config.get("model", {}).get("path"),
    }
    config_path = adapter_dir / "adapter_config.json"
    config_path.write_text(json.dumps(adapter_config, indent=2))

    # Write README.md
    model_card = generate_model_card(config, base_model=config.get("model", {}).get("path"))
    (adapter_dir / "README.md").write_text(model_card)

    return push_to_hub(
        adapter_dir,
        repo_id,
        adapter_only=True,
        private=private,
        token=token,
    )
