# Config Reference

MLX Forge uses YAML configuration files. All fields use Pydantic v2 validation with `extra="forbid"`.

## Full Example

```yaml
schema_version: 1

model:
  path: Qwen/Qwen3-0.6B
  tokenizer_path: null          # Optional separate tokenizer
  trust_remote_code: false
  revision: null                # HF revision/commit hash
  quantization:                 # Optional QLoRA
    bits: 4
    group_size: 64

adapter:
  method: lora                  # lora, dora, or full
  preset: attention-qv          # or targets: ["*.q_proj"]
  rank: 8
  scale: 20.0
  dropout: 0.0
  num_layers: null              # Apply to last N layers only

data:
  train: train.jsonl
  valid: valid.jsonl
  # OR: hf_dataset: tatsu-lab/alpaca
  max_seq_length: 2048
  mask_prompt: true
  packing: false

training:
  batch_size: 2
  num_iters: 1000
  learning_rate: 1e-5
  optimizer: adam                # adam, adamw, sgd, adafactor
  training_type: sft            # sft, dpo, grpo
  gradient_checkpointing: false
  steps_per_report: 10
  steps_per_eval: 200
  steps_per_save: 100
  seed: 42

runtime:
  run_dir: ~/.mlxforge/runs
  eager: false
```

## Sections

### `model`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | (required) | HuggingFace model ID or local path |
| `tokenizer_path` | string | null | Separate tokenizer path |
| `trust_remote_code` | bool | false | Trust remote code |
| `revision` | string | null | HF revision hash |
| `quantization.bits` | int | 4 | Quantization bits (4 or 8) |
| `quantization.group_size` | int | 64 | Quantization group size |

### `adapter`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | string | "lora" | Training method: lora, dora, full |
| `targets` | list | null | Glob patterns for target modules |
| `preset` | string | null | Named preset (attention-qv, attention-all, mlp, all-linear) |
| `rank` | int | 8 | LoRA/DoRA rank |
| `scale` | float | 20.0 | LoRA/DoRA scale factor |
| `dropout` | float | 0.0 | LoRA dropout rate |
| `num_layers` | int | null | Apply to last N layers only |

### `data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train` | string | null | Training data JSONL path |
| `valid` | string | null | Validation data JSONL path |
| `hf_dataset` | string | null | HuggingFace dataset ID |
| `hf_split` | string | "train" | HF dataset split |
| `hf_subset` | string | null | HF dataset subset |
| `hf_max_samples` | int | null | Limit samples from HF |
| `max_seq_length` | int | 2048 | Maximum sequence length |
| `mask_prompt` | bool | true | Mask prompt tokens from loss |
| `packing` | bool | false | Enable sequence packing |

### `training`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | int | 2 | Batch size |
| `num_iters` | int | 1000 | Total training iterations |
| `learning_rate` | float | 1e-5 | Learning rate |
| `optimizer` | string | "adam" | Optimizer (adam, adamw, sgd, adafactor) |
| `training_type` | string | "sft" | Training type (sft, dpo, grpo) |
| `gradient_checkpointing` | bool | false | Enable gradient checkpointing |
| `grad_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `max_grad_norm` | float | null | Gradient clipping norm |
| `seed` | int | 42 | Random seed |
| `steps_per_report` | int | 10 | Steps between metric reports |
| `steps_per_eval` | int | 200 | Steps between evaluations |
| `steps_per_save` | int | 100 | Steps between checkpoints |
| `val_batches` | int | 25 | Number of validation batches |

#### GRPO-Specific Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `grpo_num_generations` | int | 4 | Completions per prompt |
| `grpo_beta` | float | 0.1 | KL penalty coefficient |
| `grpo_clip_range` | float | 0.2 | PPO clip range |
| `grpo_max_completion_length` | int | 256 | Max completion tokens |
| `grpo_reward_function` | string | "length" | Reward function name |
