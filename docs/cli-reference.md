# CLI Reference

## `mlx-forge prepare`

Pre-tokenize a dataset for training.

```bash
mlx-forge prepare --data data.jsonl --model Qwen/Qwen3-0.6B
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | (required) | Path to JSONL data file |
| `--model` | (required) | HuggingFace model ID or local path |
| `--name` | filename | Dataset name for the registry |
| `--max-seq-length` | 2048 | Maximum sequence length |
| `--no-mask-prompt` | false | Don't mask prompt tokens from loss |
| `--trust-remote-code` | false | Trust remote code for tokenizer |

## `mlx-forge train`

Run training from a config file.

```bash
mlx-forge train --config config.yaml
mlx-forge train --config config.yaml --resume ~/.mlxforge/runs/<id>/checkpoints/step-0000500
```

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config file |
| `--resume` | Checkpoint directory to resume from |

## `mlx-forge generate`

Generate text with an optional adapter.

```bash
mlx-forge generate --model Qwen/Qwen3-0.6B --prompt "Hello"
mlx-forge generate --model Qwen/Qwen3-0.6B --adapter path/to/checkpoint
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Model ID or path |
| `--adapter` | none | Checkpoint directory with adapters |
| `--prompt` | none | Text prompt (omit for chat mode) |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--max-tokens` | 512 | Max tokens to generate |
| `--repetition-penalty` | 1.0 | Repetition penalty |
| `--seed` | none | RNG seed |

## `mlx-forge serve`

Start an OpenAI-compatible API server.

```bash
mlx-forge serve --model Qwen/Qwen3-0.6B --port 8000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | none | Model to pre-load |
| `--adapter` | none | Adapter path |
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 8000 | Port number |

## `mlx-forge export`

Merge adapters and export model.

```bash
mlx-forge export --run-id <id>
mlx-forge export --run-id <id> --format gguf
mlx-forge export --run-id <id> --push-to-hub user/model
```

| Flag | Default | Description |
|------|---------|-------------|
| `--run-id` | (required) | Run ID to export |
| `--output-dir` | auto | Output directory |
| `--checkpoint` | best/latest | Checkpoint name |
| `--format` | safetensors | Export format (safetensors, gguf) |
| `--push-to-hub` | none | HuggingFace repo ID |
| `--adapter-only` | false | Upload only adapter files |
| `--private` | false | Create private HF repo |

## `mlx-forge data`

Dataset management commands.

```bash
mlx-forge data catalog                          # Show curated catalog
mlx-forge data download alpaca-cleaned           # Download from catalog
mlx-forge data hf-import tatsu-lab/alpaca        # Import from HuggingFace
mlx-forge data import data.jsonl --name mydata   # Import local file
mlx-forge data list                              # List downloaded datasets
mlx-forge data inspect mydata --n 5              # Preview samples
mlx-forge data stats mydata                      # Show statistics
mlx-forge data validate data.jsonl               # Validate format
mlx-forge data delete mydata                     # Delete dataset
```

## `mlx-forge studio`

Launch the browser-based Studio UI.

```bash
mlx-forge studio
mlx-forge studio --port 8741
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 8741 | Port number |
