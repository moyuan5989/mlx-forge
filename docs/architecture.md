# Adding a New Architecture

MLX Forge supports adding custom model architectures. Follow this guide to add support for a new model.

## 1. Create the Architecture Module

Create `mlx_forge/models/architectures/newmodel.py`:

```python
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass

from mlx_forge.models._base.base_args import BaseModelArgs


@dataclass
class NewModelArgs(BaseModelArgs):
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    vocab_size: int = 32000

    @classmethod
    def from_dict(cls, config: dict) -> "NewModelArgs":
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class NewModel(nn.Module):
    def __init__(self, args: NewModelArgs):
        super().__init__()
        # Build model layers...

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        # inputs: (B, T) -> logits: (B, T, vocab_size)
        ...
```

## 2. Register the Architecture

In `mlx_forge/models/registry.py`:

```python
SUPPORTED_ARCHITECTURES["newmodel"] = "mlx_forge.models.architectures.newmodel"
```

## 3. Add Weight Mapping (for GGUF export)

In `mlx_forge/export/weight_mapping.py`:

```python
NEWMODEL_WEIGHT_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    # ... layer mappings
}
WEIGHT_MAPS["newmodel"] = NEWMODEL_WEIGHT_MAP
```

## 4. Add Tests

In `tests/test_model_loading.py`, add test cases for the new architecture.

## Interface Requirements

- `Model.__call__(inputs, cache=None)` must accept `(B, T)` input IDs and return `(B, T, vocab_size)` logits
- Cache objects must have `.offset` property and `.update_and_fetch(keys, values)` method
- `ModelArgs.from_dict(config)` must construct args from HuggingFace `config.json`
