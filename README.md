# MLX Forge

**Fine-tune LLMs on your Mac with MLX. No cloud, no CUDA required.**

[![PyPI](https://img.shields.io/pypi/v/mlx-forge)](https://pypi.org/project/mlx-forge/)
[![Python](https://img.shields.io/pypi/pyversions/mlx-forge)](https://pypi.org/project/mlx-forge/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/moyuan5989/mlx-forge/test.yml?label=tests)](https://github.com/moyuan5989/mlx-forge/actions)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://moyuan5989.github.io/mlx-forge)

MLX Forge is a complete LLM fine-tuning toolkit that runs entirely on your Mac. Pick a model, upload your data, and start training — all from a browser-based UI or CLI. Supports LoRA, DoRA, Full FT, QLoRA, DPO, GRPO, 18+ models, 200k+ HuggingFace datasets, OpenAI-compatible serving, and GGUF export.

```bash
pip install mlx-forge
mlx-forge studio
```

<p align="center">
  <img src="assets/studio-experiment-detail.png" alt="MLX Forge Studio — Experiment Detail" width="800">
</p>

## Why MLX Forge?

- **One command to start** — `pip install mlx-forge && mlx-forge studio`.
- **Browser-based Studio UI** — Guided training wizard, real-time loss charts, model library with memory estimates, interactive playground, one-click HuggingFace upload.
- **OpenAI-compatible API** — `mlx-forge serve` starts a server that works with Cursor, Continue.dev, Open WebUI, LangChain, and any OpenAI SDK client.
- **Runs on Apple Silicon** — Built on [MLX](https://github.com/ml-explore/mlx). Your data stays on your machine.
- **Production training features** — LoRA, DoRA, Full FT, QLoRA (67% memory reduction), DPO alignment, GRPO reinforcement learning, sequence packing (2-5x speedup), gradient checkpointing.
- **Full ecosystem** — HuggingFace datasets (200k+), Hub upload, GGUF export for Ollama/llama.cpp.

## Quick Start

### Studio UI (recommended)

```bash
mlx-forge studio
# Opens at http://127.0.0.1:8741
```

Pick a recipe, choose a model, upload your data, and start training — all from the browser.

### CLI

```bash
# Browse and download a dataset
mlx-forge data catalog
mlx-forge data download alpaca-cleaned --max-samples 5000

# Or import from HuggingFace (200k+ datasets)
mlx-forge data hf-import tatsu-lab/alpaca --max-samples 5000

# Train
mlx-forge train --config train.yaml

# Serve with OpenAI-compatible API
mlx-forge serve --model Qwen/Qwen3-0.6B --port 8000

# Export and deploy
mlx-forge export --run-id <id> --format gguf
mlx-forge export --run-id <id> --push-to-hub username/my-model
```

Models are downloaded from Hugging Face on first run and cached locally. All subsequent runs work offline.

## Studio UI

<p align="center">
  <img src="assets/studio-model-library.png" alt="MLX Forge Studio — Model Library" width="800">
</p>

- **New Training** — Guided wizard: pick a recipe (chat, instruction, DPO, writing style), choose a model, configure, and launch
- **Model Library** — Browse 18+ models with memory estimates for your hardware
- **Experiments** — Compare runs, view loss curves in real time, export and push to Hub
- **Datasets** — Manage your training data, import from HuggingFace Hub
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

**Training Methods**
- **LoRA** and **QLoRA** (4-bit) — Low-rank adaptation with 67% memory reduction
- **DoRA** — Weight-Decomposed Low-Rank Adaptation for better fine-tuning quality
- **Full Fine-Tuning** — All parameters trainable for small models
- **DPO** — Direct Preference Optimization for alignment
- **GRPO** — Group Relative Policy Optimization (DeepSeek-R1 style RL training)

**Training Features**
- Sequence packing for 2-5x speedup on short sequences
- Gradient checkpointing for 40-60% memory savings
- Compiled training loop with gradient accumulation
- Cosine, linear, step, and exponential LR schedules with warmup
- Resume from any checkpoint

**Data**
- 20+ curated datasets across 7 categories (general, code, math, conversation, reasoning, safety, domain)
- 200k+ HuggingFace datasets via `hf_dataset` config or `mlx-forge data hf-import`
- Auto-detection of 7 formats: chat, completions, text, preference, Alpaca, ShareGPT, Q&A
- Multi-dataset mixing with weighted sampling
- Data validation with train/val overlap detection

**Serving & Export**
- OpenAI-compatible API server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
- GGUF export for Ollama/llama.cpp deployment
- One-command HuggingFace Hub upload with auto-generated model cards

## CLI Reference

| Command | Description |
|---------|-------------|
| `mlx-forge studio` | Launch the Studio UI |
| `mlx-forge train --config FILE` | Run training (LoRA/DoRA/Full/DPO/GRPO) |
| `mlx-forge generate --model MODEL` | Generate text or interactive chat |
| `mlx-forge serve --model MODEL` | Start OpenAI-compatible API server |
| `mlx-forge export --run-id ID` | Export merged model (safetensors or GGUF) |
| `mlx-forge prepare --data FILE --model MODEL` | Pre-tokenize a dataset |
| `mlx-forge data catalog` | Browse 20+ curated datasets |
| `mlx-forge data download DATASET` | Download a dataset from the catalog |
| `mlx-forge data hf-import DATASET` | Import from HuggingFace Hub |
| `mlx-forge data import FILE --name NAME` | Import a local JSONL file |
| `mlx-forge data validate FILE` | Validate JSONL data |
| `mlx-forge data inspect NAME` | Preview dataset samples |
| `mlx-forge data stats NAME` | Show dataset statistics |

## Configuration

```yaml
schema_version: 1

model:
  path: "Qwen/Qwen3-0.6B"         # HF model ID or local path
  quantization:                     # Optional: QLoRA (67% memory savings)
    bits: 4
    group_size: 64

adapter:
  method: lora                      # lora | dora | full
  preset: "attention-qv"            # attention-qv | attention-all | mlp | all-linear
  rank: 16
  scale: 32.0

data:
  train: "./train.jsonl"
  valid: "./val.jsonl"
  # OR: hf_dataset: "tatsu-lab/alpaca"  # Load from HuggingFace
  packing: false                    # Sequence packing (2-5x speedup)
  max_seq_length: 2048

training:
  training_type: sft                # sft | dpo | grpo
  optimizer: adamw                  # adam | adamw | sgd | adafactor
  learning_rate: 1.0e-5
  num_iters: 1000
  batch_size: 4
  gradient_checkpointing: false     # 40-60% memory savings

runtime:
  seed: 42
```

## OpenAI-Compatible API

```bash
mlx-forge serve --model Qwen/Qwen3-0.6B --port 8000
```

Works with any OpenAI SDK client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Data Formats

MLX Forge auto-detects JSONL formats:

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
from mlx_forge import prepare, train
from mlx_forge.config import TrainingConfig

# Train from a config file
config = TrainingConfig.from_yaml("train.yaml")
result = train(config=config)
print(f"Best val loss: {result.best_val_loss:.4f}")
```

```python
from mlx_forge import generate

# Generate text with a fine-tuned adapter
generate(
    model="Qwen/Qwen3-0.6B",
    adapter="~/.mlxforge/runs/my-run/checkpoints/best",
    prompt="Explain quantum computing in simple terms.",
)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and how to submit changes.

## License

[MIT](LICENSE)
