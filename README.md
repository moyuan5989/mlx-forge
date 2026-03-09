# CortexLab

**LoRA fine-tuning for MLX on Apple Silicon — with a browser-based Studio UI.**

[![PyPI](https://img.shields.io/pypi/v/cortexlab)](https://pypi.org/project/cortexlab/)
[![Python](https://img.shields.io/pypi/pyversions/cortexlab)](https://pypi.org/project/cortexlab/)
[![License](https://img.shields.io/github/license/moyuan5989/CortexLab)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/moyuan5989/CortexLab/test.yml?label=tests)](https://github.com/moyuan5989/CortexLab/actions)

CortexLab is a framework for fine-tuning large language models on your Mac. It supports LoRA, QLoRA, DPO, sequence packing, and gradient checkpointing — all running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). A built-in browser UI (Studio) lets you launch training runs, monitor loss curves in real time, and test your models interactively.

## Features

**Training**
- LoRA and QLoRA (4-bit) fine-tuning with 67% memory reduction
- DPO (Direct Preference Optimization) for alignment training
- Sequence packing for 2-5x speedup on short sequences
- Gradient checkpointing for 40-60% activation memory savings
- Compiled training loop with gradient accumulation
- Cosine, linear, step, and exponential LR schedules with warmup
- Resume from any checkpoint

**Models**
- Llama 2/3 (all sizes)
- Mistral (mapped to Llama architecture)
- Qwen 2/3/3.5
- Phi-3/4
- Gemma 1/2/3 (1B-27B)
- Automatic Hugging Face model downloading and caching

**Studio UI**
- Browser-based training dashboard
- Real-time loss curves via WebSocket
- Model library and dataset browser
- Interactive playground for testing fine-tuned models

**Data**
- 20+ curated datasets across 7 categories (general, code, math, conversation, reasoning, safety, domain)
- Auto-detection of chat, completions, text, and preference formats
- Multi-dataset mixing with weighted sampling
- Data validation with train/val overlap detection

## Installation

```bash
pip install cortexlab
```

Requires macOS with Apple Silicon (M1/M2/M3/M4) and Python 3.10+.

## Quick Start

**1. Install and download a dataset:**

```bash
pip install cortexlab
cortexlab data catalog
cortexlab data download alpaca-cleaned --max-samples 5000
```

**2. Create a config file** (`train.yaml`):

```yaml
schema_version: 1

model:
  path: "Qwen/Qwen3-0.6B"

adapter:
  preset: "attention-qv"
  rank: 8
  scale: 16.0

data:
  train: "~/.cortexlab/datasets/raw/alpaca-cleaned/data.jsonl"
  valid: "~/.cortexlab/datasets/raw/alpaca-cleaned/data.jsonl"
  max_seq_length: 512

training:
  batch_size: 4
  num_iters: 500
  learning_rate: 1.0e-4
  optimizer: adamw
  steps_per_save: 100
  steps_per_eval: 100
  steps_per_report: 10
```

**3. Train:**

```bash
cortexlab train --config train.yaml
```

CortexLab downloads the model from Hugging Face on first run and caches it locally. All subsequent runs work offline.

## Studio UI

Launch the browser-based dashboard:

```bash
cortexlab studio
# Opens at http://127.0.0.1:8741
```

Studio provides:
- **Dashboard** — Start new training runs, monitor active jobs
- **Runs** — Browse past runs, compare loss curves
- **Models** — View downloaded models, check sizes and architectures
- **Datasets** — Browse and manage your datasets
- **Playground** — Chat with your fine-tuned models interactively

## CLI Reference

| Command | Description |
|---------|-------------|
| `cortexlab train --config FILE` | Run LoRA/QLoRA/DPO training |
| `cortexlab generate --model MODEL` | Generate text (or interactive chat without `--prompt`) |
| `cortexlab prepare --data FILE --model MODEL` | Pre-tokenize a dataset |
| `cortexlab studio` | Launch the browser-based Studio UI |
| `cortexlab data catalog` | Browse 20+ curated datasets |
| `cortexlab data download DATASET` | Download a dataset from the catalog |
| `cortexlab data import FILE --name NAME` | Import a local JSONL file |
| `cortexlab data inspect NAME` | Preview dataset samples |
| `cortexlab data stats NAME` | Show dataset statistics |
| `cortexlab data validate FILE` | Validate JSONL format and check for issues |
| `cortexlab data list` | List downloaded datasets |
| `cortexlab data delete NAME` | Delete a dataset |

## Library API

All CLI commands are backed by Python functions:

```python
from cortexlab import prepare, train
from cortexlab.config import TrainingConfig

# Pre-tokenize a dataset
prepare(data_path="train.jsonl", model="Qwen/Qwen3-0.6B")

# Train from a config file
config = TrainingConfig.from_yaml("train.yaml")
result = train(config=config)
print(f"Best val loss: {result.best_val_loss:.4f}")
```

```python
from cortexlab import generate

# Generate text with a fine-tuned adapter
generate(
    model="Qwen/Qwen3-0.6B",
    adapter="~/.cortexlab/runs/my-run/checkpoints/best",
    prompt="Explain quantum computing in simple terms.",
    temperature=0.7,
    max_tokens=256,
)
```

## Supported Models

| Architecture | Model Families | Sizes |
|-------------|---------------|-------|
| Llama | Llama 2, Llama 3, Llama 3.1, Llama 3.2 | 1B - 70B |
| Mistral | Mistral 7B, Mistral Nemo | 7B - 12B |
| Qwen | Qwen 2, Qwen 2.5, Qwen 3, Qwen 3.5 | 0.6B - 72B |
| Phi | Phi-3, Phi-3.5, Phi-4 | 3.8B - 14B |
| Gemma | Gemma 1, Gemma 2, Gemma 3 | 1B - 27B |

Models are auto-downloaded from Hugging Face on first use. Use any HF model ID (e.g., `meta-llama/Llama-3.2-1B`, `Qwen/Qwen3-0.6B`, `google/gemma-3-1b`).

## Configuration

Full training config with all options:

```yaml
schema_version: 1

model:
  path: "Qwen/Qwen3-0.6B"         # HF model ID or local path
  revision: "abc123"                # Optional: pin to specific HF commit
  quantization:                     # Optional: QLoRA (67% memory savings)
    bits: 4
    group_size: 64

adapter:
  preset: "attention-qv"           # attention-qv | attention-all | mlp | all-linear
  # targets: ["*.q_proj", "*.v_proj"]  # Or use custom glob patterns
  rank: 16
  scale: 32.0
  num_layers: 16                    # Optional: apply to last N layers only

data:
  train: "./train.jsonl"
  valid: "./val.jsonl"
  packing: false                    # Sequence packing (2-5x speedup)
  max_seq_length: 2048
  # sources:                        # Multi-dataset mixing
  #   - name: "dataset-a"
  #     weight: 0.7
  #   - name: "dataset-b"
  #     weight: 0.3

training:
  optimizer: adamw                  # adam | adamw | sgd | adafactor
  learning_rate: 1.0e-5
  num_iters: 1000
  batch_size: 4
  grad_accumulation_steps: 1
  gradient_checkpointing: false     # 40-60% memory savings (slight overhead)
  steps_per_save: 100
  steps_per_eval: 200
  steps_per_report: 10
  max_grad_norm: 1.0
  # training_type: dpo              # For DPO training
  # dpo_beta: 0.1

runtime:
  run_dir: "~/.cortexlab/runs"
  seed: 42
```

## Data Formats

CortexLab auto-detects four JSONL formats:

**Chat** — Multi-turn conversations (loss computed on assistant turns only):
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

**Completions** — Prompt-completion pairs:
```json
{"prompt": "Translate to French: Hello", "completion": "Bonjour"}
```

**Text** — Raw text for continued pretraining:
```json
{"text": "The quick brown fox jumps over the lazy dog."}
```

**Preference** — For DPO alignment training:
```json
{"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "good"}], "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "bad"}]}
```

## Advanced Features

### QLoRA (4-bit Quantization)

Reduce memory usage by ~67% with minimal quality loss:

```yaml
model:
  path: "meta-llama/Llama-3.2-3B"
  quantization:
    bits: 4
    group_size: 64
```

### Sequence Packing

Pack multiple short sequences into a single batch for 2-5x speedup:

```yaml
data:
  packing: true
  max_seq_length: 2048
```

### Gradient Checkpointing

Trade compute for memory — saves 40-60% activation memory:

```yaml
training:
  gradient_checkpointing: true
```

### DPO Training

Train with Direct Preference Optimization using preference data:

```yaml
training:
  training_type: dpo
  dpo_beta: 0.1

data:
  train: "./preference_data.jsonl"  # Must use preference format
```

### Multi-Dataset Mixing

Combine multiple datasets with weighted sampling:

```yaml
data:
  sources:
    - name: "general-chat"
      weight: 0.6
    - name: "code-instruct"
      weight: 0.4
  max_seq_length: 2048
```

### Data Validation

Check your data for issues before training:

```bash
cortexlab data validate train.jsonl --val val.jsonl
```

### Resume Training

Resume from any checkpoint:

```bash
cortexlab train --config train.yaml --resume ~/.cortexlab/runs/{run_id}/checkpoints/step-0001000
```

## Run Artifacts

Every training run produces structured artifacts:

```
~/.cortexlab/runs/{run_id}/
├── config.yaml              # Frozen config snapshot
├── manifest.json            # Run metadata + model resolution
├── environment.json         # Environment info
├── checkpoints/
│   ├── step-0000100/
│   │   ├── adapters.safetensors
│   │   ├── optimizer.safetensors
│   │   └── state.json
│   └── best -> step-0000500
└── logs/
    └── metrics.jsonl
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and how to submit changes.

## License

[MIT](LICENSE)
