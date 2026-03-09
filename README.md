# CortexLab

**Fine-tune LLMs on your Mac with MLX. No cloud, no CUDA required.**

[![PyPI](https://img.shields.io/pypi/v/cortexlab)](https://pypi.org/project/cortexlab/)
[![Python](https://img.shields.io/pypi/pyversions/cortexlab)](https://pypi.org/project/cortexlab/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/moyuan5989/CortexLab/test.yml?label=tests)](https://github.com/moyuan5989/CortexLab/actions)

CortexLab is a complete LLM fine-tuning toolkit that runs entirely on your Mac. Pick a model, upload your data, and start training — all from a browser-based UI. Supports LoRA, QLoRA, DPO, 18+ models, and 20+ curated datasets out of the box.

```bash
pip install cortexlab
cortexlab studio
```

<p align="center">
  <img src="assets/studio-new-training.png" alt="CortexLab Studio — New Training" width="800">
</p>

## Why CortexLab?

- **One command to start** — `pip install cortexlab && cortexlab studio`.
- **Browser-based Studio UI** — Guided training wizard, real-time loss charts, model library with memory estimates, interactive playground.
- **Runs on Apple Silicon** — Built on [MLX](https://github.com/ml-explore/mlx). Your data stays on your machine.
- **Production training features** — QLoRA (67% memory reduction), sequence packing (2-5x speedup), gradient checkpointing, DPO alignment, compiled training loop.

## Quick Start

### Studio UI (recommended)

```bash
cortexlab studio
# Opens at http://127.0.0.1:8741
```

Pick a recipe, choose a model, upload your data, and start training — all from the browser.

### CLI

```bash
# Browse and download a dataset
cortexlab data catalog
cortexlab data download alpaca-cleaned --max-samples 5000

# Train
cortexlab train --config train.yaml
```

Models are downloaded from Hugging Face on first run and cached locally. All subsequent runs work offline.

## Studio UI

<p align="center">
  <img src="assets/studio-model-library.png" alt="CortexLab Studio — Model Library" width="800">
</p>

- **New Training** — Guided wizard: pick a recipe (chat, instruction, DPO, writing style), choose a model, configure, and launch
- **Model Library** — Browse 18+ models with memory estimates for your hardware
- **Experiments** — Compare runs, view loss curves in real time
- **Datasets** — Manage your training data
- **Playground** — Chat with your fine-tuned models interactively

## Supported Models

18 curated models in the Studio library, all tested on Apple Silicon:

| Architecture | Models | Sizes |
|-------------|--------|-------|
| Qwen | Qwen 2.5, Qwen 3, Qwen 3.5 | 0.5B - 8B |
| Gemma | Gemma 2, Gemma 3 | 1B - 9B |
| Llama | Llama 3.1 | 8B |
| Phi | Phi-3 Mini, Phi-4 Mini | 3.8B |
| DeepSeek | DeepSeek-R1-Distill (Qwen-based) | 1.5B - 7B |
| Mistral | Mistral (uses Llama architecture) | 7B |

Any HF model using a supported architecture will work — the table above shows the curated models with pre-computed memory estimates in Studio.

## Features

**Training**
- LoRA and QLoRA (4-bit) fine-tuning with 67% memory reduction
- DPO (Direct Preference Optimization) for alignment
- Sequence packing for 2-5x speedup on short sequences
- Gradient checkpointing for 40-60% memory savings
- Compiled training loop with gradient accumulation
- Cosine, linear, step, and exponential LR schedules with warmup
- Resume from any checkpoint

**Data**
- 20+ curated datasets across 7 categories (general, code, math, conversation, reasoning, safety, domain)
- Auto-detection of chat, completions, text, and preference formats
- Multi-dataset mixing with weighted sampling
- Data validation with train/val overlap detection

## CLI Reference

| Command | Description |
|---------|-------------|
| `cortexlab studio` | Launch the Studio UI |
| `cortexlab train --config FILE` | Run LoRA/QLoRA/DPO training |
| `cortexlab generate --model MODEL` | Generate text or interactive chat |
| `cortexlab prepare --data FILE --model MODEL` | Pre-tokenize a dataset |
| `cortexlab data catalog` | Browse 20+ curated datasets |
| `cortexlab data download DATASET` | Download a dataset from the catalog |
| `cortexlab data import FILE --name NAME` | Import a local JSONL file |
| `cortexlab data validate FILE` | Validate JSONL data |
| `cortexlab data inspect NAME` | Preview dataset samples |
| `cortexlab data stats NAME` | Show dataset statistics |

## Configuration

```yaml
schema_version: 1

model:
  path: "Qwen/Qwen3-0.6B"         # HF model ID or local path
  quantization:                     # Optional: QLoRA (67% memory savings)
    bits: 4
    group_size: 64

adapter:
  preset: "attention-qv"           # attention-qv | attention-all | mlp | all-linear
  rank: 16
  scale: 32.0

data:
  train: "./train.jsonl"
  valid: "./val.jsonl"
  packing: false                    # Sequence packing (2-5x speedup)
  max_seq_length: 2048

training:
  optimizer: adamw                  # adam | adamw | sgd | adafactor
  learning_rate: 1.0e-5
  num_iters: 1000
  batch_size: 4
  gradient_checkpointing: false     # 40-60% memory savings
  # training_type: dpo              # For DPO training
  # dpo_beta: 0.1

runtime:
  seed: 42
```

## Data Formats

CortexLab auto-detects four JSONL formats:

**Chat** — Multi-turn conversations (loss on assistant turns only):
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

## Library API

All CLI commands are backed by Python functions:

```python
from cortexlab import prepare, train
from cortexlab.config import TrainingConfig

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
)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and how to submit changes.

## License

[MIT](LICENSE)
