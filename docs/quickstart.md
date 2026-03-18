# Quick Start

Train your first model in 5 minutes.

## 1. Download a Dataset

```bash
mlx-forge data download alpaca-cleaned --max-samples 1000
```

Or import from HuggingFace:

```bash
mlx-forge data hf-import tatsu-lab/alpaca --max-samples 1000 --name alpaca
```

## 2. Create a Config

Create `config.yaml`:

```yaml
schema_version: 1

model:
  path: Qwen/Qwen3-0.6B

adapter:
  method: lora          # or "dora", "full"
  preset: attention-qv
  rank: 8
  scale: 20.0

data:
  train: ~/.mlxforge/datasets/raw/alpaca_train.jsonl
  valid: ~/.mlxforge/datasets/raw/alpaca_val.jsonl
  max_seq_length: 2048
  mask_prompt: true

training:
  batch_size: 2
  num_iters: 500
  learning_rate: 1e-5
  optimizer: adam
  steps_per_report: 10
  steps_per_eval: 100
  steps_per_save: 100

runtime:
  run_dir: ~/.mlxforge/runs
```

## 3. Train

```bash
mlx-forge train --config config.yaml
```

## 4. Generate

```bash
mlx-forge generate \
  --model Qwen/Qwen3-0.6B \
  --adapter ~/.mlxforge/runs/<run-id>/checkpoints/best \
  --prompt "Explain quantum computing"
```

## 5. Export & Deploy

```bash
# Export as safetensors
mlx-forge export --run-id <run-id>

# Export as GGUF for Ollama
mlx-forge export --run-id <run-id> --format gguf

# Push to HuggingFace
mlx-forge export --run-id <run-id> --push-to-hub username/my-model
```

## 6. Serve with OpenAI API

```bash
mlx-forge serve --model <run-id> --port 8000
```

Then use with any OpenAI-compatible client:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<run-id>",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Next Steps

- [Launch Studio UI](studio.md) for a visual training experience
- [CLI Reference](cli-reference.md) for all commands
- [Config Reference](config-reference.md) for all configuration options
