# M8: Self-Contained Model Loading Design

> **Goal**: Remove the `mlx-lm` dependency and make LMForge fully self-contained for model loading.

---

## 1. Executive Summary

**Problem**: LMForge currently depends on `mlx-lm` for model loading. This is problematic because:
1. `mlx-lm` is a demo/example library, not production-grade
2. Unstable API with no semver guarantees
3. We only need ~5% of its functionality (model loading, not generation/serving)
4. Adds uncertainty and dependency risk to LMForge's stability

**Solution**: Vendor the minimal necessary components into LMForge:
- Model architecture implementations (Llama, Qwen, Phi, Gemma families)
- Weight loading utilities
- Config parsing
- Registry pattern for model discovery

**Scope**: Training/fine-tuning only. No generation, no inference serving.

---

## 2. Research Findings

### 2.1 mlx-lm Architecture Analysis

**Loading Flow** (`mlx_lm.utils.load()`):
```
load() → _download() → load_model() → load_tokenizer()
               ↓
        load_config()
               ↓
        _get_classes() → importlib.import_module(f"mlx_lm.models.{model_type}")
               ↓
        model = Model(ModelArgs.from_dict(config))
               ↓
        model.sanitize(weights)
               ↓
        model.load_weights(weights)
```

**Key Components**:
1. **Registry** (`_get_classes()`): Dynamic import based on `model_type` in config.json
2. **Model Remapping**: `MODEL_REMAPPING` dict maps aliases (e.g., `mistral` → `llama`)
3. **BaseModelArgs**: Provides `from_dict()` that filters config to dataclass fields
4. **Weight Sanitization**: `model.sanitize()` remaps/filters weight keys before loading
5. **Weight Loading**: Uses `mx.load()` for safetensors, then `model.load_weights()`

**Model Structure Pattern** (consistent across all architectures):
```python
@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    # ... architecture-specific fields

class Attention(nn.Module): ...
class MLP(nn.Module): ...
class TransformerBlock(nn.Module): ...
class {Name}Model(nn.Module): ...  # Backbone

class Model(nn.Module):  # Required export
    def __init__(self, args: ModelArgs): ...
    def __call__(self, inputs, cache=None): ...
    def sanitize(self, weights): ...  # Optional
    @property
    def layers(self): return self.model.layers
```

### 2.2 Model Complexity Analysis

| Architecture | Lines | Complexity | Notes |
|-------------|-------|------------|-------|
| llama.py | 275 | Medium | Template for Llama, Mistral |
| qwen3.py | 224 | Low | Simple, no sliding window |
| phi3.py | 214 | Medium | Fused QKV proj, SuScaled RoPE |
| gemma2.py | 206 | Medium | Custom RMSNorm, logit softcapping |
| deepseek_v3.py | 499 | High | MoE, complex weight transforms |

**Observation**: Most popular models share 90% structure. Differences are in:
- Attention variants (GQA, sliding window)
- MLP variants (SwiGLU, GeGLU, fused projections)
- Normalization (RMSNorm with +1 offset for Gemma)
- RoPE variants (standard, Llama3, SuScaled, Yarn)

### 2.3 Shared Utilities

**Required from mlx-lm**:
- `base.py`: `BaseModelArgs`, `create_attention_mask()`, `scaled_dot_product_attention()`
- `rope_utils.py`: `initialize_rope()`, `Llama3RoPE`, `SuScaledRoPE`, `YarnRoPE`
- `activations.py`: `swiglu()` (compiled SiLU gate)
- `cache.py`: `KVCache`, `RotatingKVCache` (for sliding window)

**NOT needed**:
- Generation utilities
- Quantization transforms (AWQ, GPTQ, bitnet)
- Distributed sharding
- Tokenizer wrapper

---

## 3. Proposed Design

### 3.1 Package Structure

```
lmforge/models/
├── __init__.py              # Public API: load_model()
├── resolve.py               # M7 - HF resolution (unchanged)
├── loader.py                # Updated: self-contained loading
├── registry.py              # Model registry + _get_classes()
│
├── _base/                   # Shared utilities (vendored, minimal)
│   ├── __init__.py
│   ├── args.py              # BaseModelArgs with from_dict()
│   ├── attention.py         # create_attention_mask(), scaled_dot_product_attention()
│   ├── rope.py              # RoPE variants: standard, Llama3, SuScaled, Yarn
│   └── activations.py       # swiglu(), gelu variants
│
└── architectures/           # Model implementations
    ├── __init__.py          # Architecture discovery
    ├── llama.py             # Llama, Mistral (via remap)
    ├── qwen3.py             # Qwen3
    ├── phi3.py              # Phi-3
    └── gemma2.py            # Gemma 2
```

### 3.2 Registry Pattern

```python
# lmforge/models/registry.py

MODEL_REMAPPING = {
    "mistral": "llama",     # Mistral uses Llama architecture
    "qwen3": "qwen3",       # Explicit mapping for clarity
    "phi3": "phi3",
    "gemma2": "gemma2",
}

SUPPORTED_ARCHITECTURES = {
    "llama": "lmforge.models.architectures.llama",
    "qwen3": "lmforge.models.architectures.qwen3",
    "phi3": "lmforge.models.architectures.phi3",
    "gemma2": "lmforge.models.architectures.gemma2",
}

def get_model_classes(config: dict) -> tuple[type, type]:
    """
    Get Model and ModelArgs classes for a given config.

    Args:
        config: Model config dict (from config.json)

    Returns:
        (Model, ModelArgs) tuple

    Raises:
        ValueError: If model_type is not supported
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)

    if model_type not in SUPPORTED_ARCHITECTURES:
        supported = sorted(SUPPORTED_ARCHITECTURES.keys())
        raise ValueError(
            f"Model type '{model_type}' is not supported. "
            f"Supported architectures: {supported}"
        )

    module_path = SUPPORTED_ARCHITECTURES[model_type]
    arch = importlib.import_module(module_path)
    return arch.Model, arch.ModelArgs
```

**Key Difference from mlx-lm**: Explicit allowlist instead of dynamic discovery. This is more secure and makes supported models explicit.

### 3.3 Updated Loader

```python
# lmforge/models/loader.py

import glob
import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
from transformers import AutoTokenizer

from .registry import get_model_classes


def load_config(model_path: Path) -> dict:
    """Load and return model config.json."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path) as f:
        return json.load(f)


def load_weights(model_path: Path) -> dict[str, mx.array]:
    """Load all safetensors weight files from model directory."""
    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))

    if not weight_files:
        raise FileNotFoundError(
            f"No safetensors weight files found in {model_path}. "
            f"Expected files matching 'model*.safetensors'"
        )

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights


def load_model(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """
    Load a model and tokenizer from a local path.

    Args:
        model_path: Local path to model directory (post-resolution)
        tokenizer_path: Optional separate tokenizer path
        trust_remote_code: Whether to trust remote code in tokenizer

    Returns:
        (model, tokenizer) tuple
    """
    model_path = Path(model_path)
    tok_path = tokenizer_path if tokenizer_path else str(model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=trust_remote_code,
    )

    # Load config and resolve model class
    config = load_config(model_path)
    model_class, model_args_class = get_model_classes(config)

    # Instantiate model
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # Load weights
    weights = load_weights(model_path)

    # Apply model-specific weight sanitization
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Load into model
    model.eval()
    model.load_weights(list(weights.items()), strict=True)

    # Force evaluation to ensure weights are loaded
    mx.eval(model.parameters())

    return model, tokenizer
```

### 3.4 Base Utilities

#### BaseModelArgs (from mlx-lm)

```python
# lmforge/models/_base/args.py

import inspect
from dataclasses import dataclass


@dataclass
class BaseModelArgs:
    """Base class for model configuration arguments."""

    @classmethod
    def from_dict(cls, params: dict):
        """
        Create ModelArgs from a config dict, filtering to valid fields.

        This allows config.json to have extra fields that aren't used.
        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
```

#### Attention Utilities

```python
# lmforge/models/_base/attention.py

from typing import Optional
import mlx.core as mx


def create_causal_mask(N: int, offset: int = 0) -> mx.array:
    """Create a causal attention mask."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    return linds[:, None] >= rinds[None]


def create_attention_mask(h: mx.array, cache=None):
    """
    Create attention mask for transformer layers.

    Args:
        h: Hidden states [B, L, D]
        cache: Optional KV cache

    Returns:
        Mask or "causal" string for optimized path
    """
    N = h.shape[1]
    if N == 1:
        return None
    return "causal"


def scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """
    Compute scaled dot-product attention.

    Uses MLX's fast SDPA implementation.
    """
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask
    )
```

#### RoPE Utilities

```python
# lmforge/models/_base/rope.py

from typing import Optional, Dict, Union, List
import math
import mlx.core as mx
import mlx.nn as nn


def initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: Optional[dict] = None,
    max_position_embeddings: Optional[int] = None,
) -> nn.Module:
    """
    Initialize the appropriate RoPE implementation based on config.

    Supports:
    - Standard RoPE
    - Linear scaling
    - Llama3 RoPE
    - SuScaled (longrope)
    - Yarn
    """
    if scaling_config is None:
        return nn.RoPE(dims, traditional=traditional, base=base)

    rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")

    if rope_type in ["default", "linear"]:
        scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)

    elif rope_type == "llama3":
        return Llama3RoPE(dims, base, traditional, scaling_config, max_position_embeddings)

    elif rope_type == "longrope":
        return SuScaledRoPE(dims, base, scaling_config, max_position_embeddings)

    elif rope_type == "yarn":
        return YarnRoPE(dims, base, traditional, scaling_config, max_position_embeddings)

    else:
        raise ValueError(f"Unsupported RoPE type: {rope_type}")


# Llama3RoPE, SuScaledRoPE, YarnRoPE implementations...
# (Vendor from mlx-lm with minimal changes)
```

### 3.5 Architecture Example: Qwen3

```python
# lmforge/models/architectures/qwen3.py
"""Qwen3 model architecture for LMForge."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .._base.rope import initialize_rope
from .._base.activations import swiglu


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # Qwen3 has QK-norm
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask=None, cache=None):
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    """Qwen3 model for LMForge training."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights: dict) -> dict:
        """Remove/remap weights before loading."""
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        """Access transformer layers for LoRA targeting."""
        return self.model.layers
```

---

## 4. What We Vendor vs. Implement

### 4.1 Vendor from mlx-lm (MIT License)

| Component | Lines | Modifications |
|-----------|-------|---------------|
| `BaseModelArgs.from_dict()` | ~10 | None |
| `create_attention_mask()` | ~15 | Simplify (remove cache quantization) |
| `scaled_dot_product_attention()` | ~30 | Simplify (remove quantized path) |
| `swiglu()` | ~5 | None |
| RoPE variants | ~200 | Consolidate into single file |
| llama.py | ~150 | Remove shard(), make_cache() |
| qwen3.py | ~150 | Remove shard() |
| phi3.py | ~150 | Remove shard() |
| gemma2.py | ~150 | Remove shard(), add custom RMSNorm |

### 4.2 Implement Ourselves

| Component | Lines | Notes |
|-----------|-------|-------|
| `registry.py` | ~50 | Explicit allowlist, not dynamic import |
| `loader.py` | ~80 | Simplified, no quantization |
| Weight loading | ~30 | Direct mx.load() |
| Config loading | ~20 | Just json.load() |

### 4.3 NOT Implementing

- KV cache for inference (training doesn't need it)
- Quantized attention (SDPA quantization)
- AWQ/GPTQ weight transforms
- Distributed sharding
- Generation sampling
- Tokenizer wrapper (use transformers directly)

---

## 5. Supported Models (Initial)

### 5.1 Tier 1: Day-One Support

| Model Family | model_type | Architecture File | Popular Models |
|--------------|------------|-------------------|----------------|
| Llama | llama | llama.py | Llama-3.2-1B/3B, Llama-3.1-8B |
| Mistral | mistral → llama | llama.py | Mistral-7B-v0.3 |
| Qwen3 | qwen3 | qwen3.py | Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B |

### 5.2 Tier 2: Follow-up

| Model Family | model_type | Notes |
|--------------|------------|-------|
| Phi-3 | phi3 | Fused QKV, SuScaled RoPE |
| Gemma 2 | gemma2 | Custom RMSNorm (+1 offset) |
| Qwen2 | qwen2 | Older Qwen architecture |

### 5.3 Future Consideration

| Model Family | Complexity | Notes |
|--------------|-----------|-------|
| DeepSeek V3 | High | MoE, complex weight transforms |
| Mixtral | Medium | MoE |
| Mamba | Medium | SSM (not transformer) |

---

## 6. Error Messages

### 6.1 Unsupported Architecture

```
ValueError: Model type 'deepseek_v3' is not supported.

Supported architectures: gemma2, llama, phi3, qwen3

If you need this architecture, please open an issue at:
https://github.com/yourusername/lmforge/issues
```

### 6.2 Missing config.json

```
FileNotFoundError: config.json not found in /path/to/model

Ensure the model directory contains:
  - config.json (model configuration)
  - model*.safetensors (model weights)
```

### 6.3 Missing Weight Files

```
FileNotFoundError: No safetensors weight files found in /path/to/model

Expected files matching 'model*.safetensors'.
Found files: config.json, tokenizer.json

The model may not have been fully downloaded. Try:
  huggingface-cli download owner/model --local-dir /path/to/model
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_model_loading.py

class TestRegistry:
    def test_known_model_type_returns_classes(self):
        config = {"model_type": "qwen3", ...}
        Model, ModelArgs = get_model_classes(config)
        assert Model is not None
        assert ModelArgs is not None

    def test_remapped_model_type(self):
        config = {"model_type": "mistral", ...}
        Model, ModelArgs = get_model_classes(config)
        # Should return llama classes

    def test_unknown_model_type_raises(self):
        config = {"model_type": "unknown_model"}
        with pytest.raises(ValueError, match="not supported"):
            get_model_classes(config)


class TestModelLoading:
    def test_load_qwen3_model(self, qwen3_model_path):
        model, tokenizer = load_model(qwen3_model_path)
        assert hasattr(model, "layers")
        assert len(model.layers) > 0

    def test_forward_pass_produces_logits(self, qwen3_model_path):
        model, tokenizer = load_model(qwen3_model_path)
        inputs = mx.array([[1, 2, 3]])
        logits = model(inputs)
        assert logits.shape[-1] == model.args.vocab_size


class TestBaseModelArgs:
    def test_from_dict_filters_unknown_keys(self):
        @dataclass
        class Args(BaseModelArgs):
            hidden_size: int
            num_layers: int

        config = {"hidden_size": 1024, "num_layers": 12, "unknown_key": "ignored"}
        args = Args.from_dict(config)
        assert args.hidden_size == 1024
        assert not hasattr(args, "unknown_key")
```

### 7.2 Integration Tests

```python
class TestEndToEnd:
    def test_load_and_apply_lora(self, qwen3_model_path):
        model, tokenizer = load_model(qwen3_model_path)

        # Apply LoRA (using existing LMForge adapter code)
        patterns = ["*.self_attn.q_proj", "*.self_attn.v_proj"]
        targets = resolve_targets(model, patterns, num_layers=None)
        apply_lora(model, targets, config)

        # Verify forward pass still works
        inputs = mx.array([[1, 2, 3]])
        logits = model(inputs)
        assert logits.shape[-1] == model.args.vocab_size

    def test_full_training_loop(self, qwen3_model_path, train_data):
        # Integration test: load model, apply LoRA, train one batch
        ...
```

---

## 8. Implementation Plan

### Phase 1: Core Infrastructure (~200 lines)
1. Create `lmforge/models/_base/` with vendored utilities
2. Implement `registry.py` with explicit allowlist
3. Update `loader.py` to use new infrastructure

### Phase 2: Tier 1 Architectures (~400 lines)
1. Port `llama.py` (handles Llama, Mistral)
2. Port `qwen3.py` (for Qwen3 family)
3. Add unit tests for each architecture

### Phase 3: Integration & Testing (~100 lines)
1. Update `lmforge/__init__.py` to use new loader
2. Integration tests with real models
3. Remove mlx-lm dependency from optional imports

### Phase 4: Tier 2 Architectures (Optional, ~300 lines)
1. Port `phi3.py`
2. Port `gemma2.py`
3. Additional tests

---

## 9. Backward Compatibility

### 9.1 No Breaking Changes

- `load_model()` function signature unchanged
- Same return type: `(model, tokenizer)` tuple
- Model weights paths unchanged (from M7 resolution)

### 9.2 Behavioral Changes

| Aspect | Before (mlx-lm) | After (self-contained) |
|--------|-----------------|------------------------|
| Supported models | All 90+ in mlx-lm | Explicit allowlist |
| Quantization | AWQ/GPTQ/bitnet | Not supported |
| Inference caching | Supported | Not needed (training only) |
| Error messages | Generic | LMForge-specific |

---

## 10. Alternatives Considered

### 10.1 Keep mlx-lm as Optional

**Rejected**: Still adds dependency risk, harder to maintain error messages, version conflicts.

### 10.2 Fork mlx-lm

**Rejected**: Maintenance burden, need to track upstream changes.

### 10.3 Abstract Model Interface

**Rejected**: Over-engineering. We only need forward pass for training.

---

## 11. Open Questions

1. **Q: Should we support quantized models for training?**
   A: No. Quantization is for inference. Training uses full precision with LoRA adapters.

2. **Q: Should we add KV cache support?**
   A: No. KV cache is for generation/inference. Training processes full sequences.

3. **Q: How do we handle new model architectures?**
   A: Add to registry when requested. Clear error message points users to issue tracker.

---

## 12. Success Criteria

1. ✅ Training works without mlx-lm installed
2. ✅ Qwen3-0.6B loads and trains successfully
3. ✅ Llama-3.2-1B loads and trains successfully
4. ✅ Error messages are clear for unsupported models
5. ✅ All existing tests pass
6. ✅ No v0 contracts broken

---

## 13. Estimated Effort

| Component | Lines | Effort |
|-----------|-------|--------|
| _base utilities | ~300 | Low (vendor) |
| registry.py | ~50 | Low |
| loader.py | ~80 | Low |
| llama.py | ~180 | Medium (adapt) |
| qwen3.py | ~180 | Medium (adapt) |
| Tests | ~150 | Medium |
| **Total** | **~940** | **Medium** |

**Timeline**: Can be implemented in a single session.
