# LMForge

**LoRA SFT Training Framework for MLX on Apple Silicon**

LMForge is a production-ready framework for fine-tuning large language models using LoRA (Low-Rank Adaptation) on Apple Silicon via MLX. It provides a simple, library-first API with automatic Hugging Face model loading, comprehensive checkpointing, and structured run management.

## Features

- **Automatic HF Model Loading** - Use `model.path: "Qwen/Qwen3-0.8B"` directly in your config
- **LoRA Fine-Tuning** - Efficient adapter-based training with glob-based targeting
- **Data Pipeline** - Auto-detection of chat/completions/text formats with caching
- **Compiled Training** - MLX-compiled training loop with gradient accumulation
- **Smart Checkpointing** - Atomic saves, retention policy, automatic resume
- **Metrics Logging** - JSONL metrics + console output + optional WandB
- **Offline Mode** - Full offline capability after first model download
- **Library-First** - All operations callable as Python functions

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lmforge.git
cd lmforge

# Install with development dependencies
pip install -e ".[dev]"

# Optional: Install mlx-lm for model loading
pip install mlx-lm
```

### Fine-Tune a Model (HF Model ID)

1. **Create a config file** (`train.yaml`):

```yaml
schema_version: 1

model:
  path: "Qwen/Qwen3-0.8B"  # Hugging Face model ID
  trust_remote_code: false

adapter:
  preset: "attention-qv"  # Apply LoRA to Q and V projections
  rank: 8
  scale: 20.0

data:
  train: "./data/train.jsonl"
  valid: "./data/valid.jsonl"

training:
  batch_size: 4
  num_iters: 1000
  learning_rate: 1.0e-5
  optimizer: adam
  steps_per_save: 100
  steps_per_eval: 200
```

2. **Prepare your data** (JSONL format):

```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well!"}]}
```

3. **Run training**:

```bash
lmforge train --config train.yaml
```

On first run, LMForge will:
- Automatically download `Qwen/Qwen3-0.8B` from Hugging Face
- Cache it locally (uses standard HF cache)
- Tokenize and cache your data
- Apply LoRA adapters
- Train the model

### Offline Mode

After the first download, training works completely offline:

```bash
export HF_HUB_OFFLINE=1
lmforge train --config train.yaml
```

### Pin a Specific Model Revision

For reproducibility, pin to a specific commit hash:

```yaml
model:
  path: "Qwen/Qwen3-0.8B"
  revision: "a1b2c3d4e5f6"  # Specific HF commit hash
```

### Using a Local Model

You can also use a local model directory:

```yaml
model:
  path: "/path/to/local/model"
```

## Library API

All CLI commands are backed by Python functions:

```python
from lmforge import prepare, train

# Prepare data (tokenization + caching)
meta = prepare(
    data_path="train.jsonl",
    model="Qwen/Qwen3-0.8B",
)

# Train from config
from lmforge.config import TrainingConfig

config = TrainingConfig.from_yaml("train.yaml")
final_state = train(config=config)

print(f"Best validation loss: {final_state.best_val_loss:.4f}")
```

## Run Artifacts

Every training run produces structured artifacts:

```
~/.lmforge/runs/{YYYYMMDD-HHMMSS-sft-Qwen3-0.8B-a3f1}/
├── config.yaml              # Frozen config snapshot
├── manifest.json            # Full run metadata + model resolution
├── environment.json         # Environment info
├── checkpoints/
│   ├── step-0000100/       # Exactly 3 files per checkpoint
│   │   ├── adapters.safetensors
│   │   ├── optimizer.safetensors
│   │   └── state.json
│   └── best -> step-0000500
└── logs/
    └── metrics.jsonl        # Training + eval metrics
```

## Resume Training

```bash
lmforge train --config train.yaml --resume ~/.lmforge/runs/{run_id}/checkpoints/step-0001000
```

## Data Formats

LMForge auto-detects three JSONL formats:

**Chat format:**
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Completions format:**
```jsonl
{"prompt": "...", "completion": "..."}
```

**Text format:**
```jsonl
{"text": "..."}
```

## Adapter Targeting

LMForge uses glob patterns to target modules for LoRA:

**Built-in presets:**
- `attention-qv`: Q and V projections only
- `attention-all`: All attention projections (Q, K, V, O)
- `mlp`: MLP layers (gate, up, down)
- `all-linear`: All attention + MLP layers

**Custom targeting:**
```yaml
adapter:
  targets:
    - "*.self_attn.q_proj"
    - "*.self_attn.v_proj"
  rank: 8
```

**Target last N layers:**
```yaml
adapter:
  preset: "attention-qv"
  num_layers: 16  # Apply to last 16 layers only
  rank: 8
```

## Model Resolution

LMForge automatically resolves Hugging Face model IDs to local paths:

1. **On first use**: Downloads from HF Hub, caches locally
2. **Subsequent uses**: Uses cached version (no network access)
3. **Revision pinning**: Records exact commit hash in manifest for reproducibility

The resolution happens **before** training starts, ensuring:
- No network access during training
- Clear error messages for auth/network issues
- Full offline capability after first download

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_resolve.py -v
```

## Architecture

LMForge follows these design principles:

- **Library-first** - All operations are Python functions; CLI is thin wrapper
- **Frozen contracts** - Config schema, batch format, checkpoint layout are immutable
- **Explicit targeting** - Glob patterns on module paths, no type-based scanning
- **Tier-1 checkpointing** - Adapters + optimizer + state (state-consistent resume)
- **Stateless LR schedules** - Pure functions of step number, no saved scheduler state
- **Fail fast** - Validate all configs before loading models or data

## License

[Add your license here]

## Contributing

[Add contribution guidelines]
