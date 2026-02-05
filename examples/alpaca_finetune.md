# Example: Fine-tuning Qwen3-0.6B on Alpaca Dataset

This example shows the complete workflow for fine-tuning a model using a HuggingFace dataset.

## Prerequisites

```bash
# Install LMForge
pip install -e .

# Install dataset conversion dependencies
pip install datasets tqdm
```

## Step 1: Download and Convert Dataset

Download the Stanford Alpaca instruction-following dataset (52K samples):

```bash
# Full dataset
python scripts/download_hf_dataset.py alpaca --output data/alpaca

# Or test with small sample first (recommended)
python scripts/download_hf_dataset.py alpaca --output data/alpaca-small --max-samples 1000
```

This creates:
- `data/alpaca/train.jsonl` (49,400 samples, 95%)
- `data/alpaca/valid.jsonl` (2,600 samples, 5%)

## Step 2: Create Training Config

Create `examples/alpaca_qwen3.yaml`:

```yaml
schema_version: 1

model:
  path: Qwen/Qwen3-0.6B
  tokenizer_path: null
  trust_remote_code: false
  revision: null

adapter:
  method: lora
  targets: null
  preset: attention-qv
  num_layers: null
  rank: 16
  scale: 32.0
  dropout: 0.0

data:
  train: data/alpaca/train.jsonl
  valid: data/alpaca/valid.jsonl
  test: null
  cache_dir: ~/.lmforge/cache/preprocessed
  max_seq_length: 2048
  mask_prompt: true

training:
  batch_size: 4
  num_iters: 10000
  learning_rate: 0.0003
  optimizer: adamw
  optimizer_config:
    weight_decay: 0.01
  lr_schedule:
    name: cosine_decay
    arguments: [0.0003, 10000, 0.000001]  # [init, decay_steps, end]
    warmup: 100
    warmup_init: 0.000001
  grad_accumulation_steps: 4
  max_grad_norm: 1.0
  seed: 42
  steps_per_report: 10
  steps_per_eval: 500
  steps_per_save: 1000
  val_batches: 50
  keep_last_n_checkpoints: 3

runtime:
  run_dir: ~/.lmforge/runs
  eager: false
  report_to: null
  wandb_project: null
```

**Note on lr_schedule**:
- Set to `null` for constant learning rate
- Or use a dictionary for scheduled learning rate:
  ```yaml
  # Cosine annealing (recommended)
  lr_schedule:
    name: cosine_decay
    arguments: [0.0003, 10000, 0.000001]  # [init, decay_steps, end]
    warmup: 100
    warmup_init: 0.000001

  # Linear decay
  lr_schedule:
    name: linear_schedule
    arguments: [0.0003, 0.000001, 10000]  # [init, end, steps]
    warmup: 100
    warmup_init: 0.000001

  # Step decay (drop by 0.5 every 2000 steps)
  lr_schedule:
    name: step_decay
    arguments: [0.0003, 0.5, 2000]  # [init, decay_rate, decay_steps]
    warmup: 100
    warmup_init: 0.000001

  # Exponential decay
  lr_schedule:
    name: exponential_decay
    arguments: [0.0003, 0.95]  # [init, decay_rate]
    warmup: 100
    warmup_init: 0.000001
  ```

## Step 3: Prepare (Optional but Recommended)

Pre-tokenize the dataset to verify format and cache it:

```bash
lmforge prepare \
  --data data/alpaca/train.jsonl \
  --model Qwen/Qwen3-0.6B \
  --output ~/.lmforge/cache/preprocessed
```

Expected output:
```
Resolving model: Qwen/Qwen3-0.6B...
  → Latest revision: c1899de2
Loading tokenizer...
Reading data/alpaca/train.jsonl...
Detected format: chat
Validating 49400 samples...
Tokenizing 49400 samples...
✓ Preprocessed 49400 samples
  Total tokens: 12,345,678
  Min/mean/max length: 15/250.3/2048
  Shards: 25
```

## Step 4: Train

Start training:

```bash
lmforge train --config examples/alpaca_qwen3.yaml
```

Expected output:
```
LMForge v0 — Training
Model: Qwen/Qwen3-0.6B
Adapter: lora (rank=16)

Resolving model...
Loading model and tokenizer...
Model loaded: Model

Applying LoRA adapters...
Matched 56 modules
Trainable parameters: 1,018,232,832 / 597,196,800 (170.51%)

Loading training data...
Cache hit: sha256:abc123...
  49400 samples, 12,345,678 tokens

Starting training...

Step 10/10000 | loss=2.891 | lr=3.00e-04 | tok/s=14521 | mem=11.2GB
Step 20/10000 | loss=2.654 | lr=2.99e-04 | tok/s=14893 | mem=11.2GB
...
Step 500/10000 | val_loss=1.987
Step 1000/10000 | Saved checkpoint
...
Training complete!
Final step: 10000
Best validation loss: 0.823
```

## Step 5: Monitor Training

View metrics:

```bash
# Real-time monitoring
tail -f ~/.lmforge/runs/*/logs/metrics.jsonl | jq

# Plot loss curve (requires matplotlib)
python -c "
import json
import matplotlib.pyplot as plt

losses = []
with open('~/.lmforge/runs/YOUR_RUN_ID/logs/metrics.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if data['event'] == 'train':
            losses.append(data['train_loss'])

plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_curve.png')
"
```

## Step 6: Use the Fine-tuned Adapter

The trained LoRA adapters are saved in:
```
~/.lmforge/runs/20260203-HHMMSS-sft-Qwen3-0.6B-XXXX/
├── checkpoints/
│   ├── step-0001000/
│   ├── step-0002000/
│   └── best -> step-NNNNNNN
└── manifest.json
```

To use the adapter for inference (using mlx-lm or other tools):

```python
import mlx.core as mx
from mlx_lm import load, generate

# Load base model
model, tokenizer = load("Qwen/Qwen3-0.6B")

# Load LoRA weights
adapter_path = "~/.lmforge/runs/.../checkpoints/best/adapters.safetensors"
model.load_weights(adapter_path, strict=False)

# Generate
prompt = "Explain quantum computing in simple terms."
response = generate(model, tokenizer, prompt, max_tokens=256)
print(response)
```

## Hyperparameter Tuning

### For Better Quality (Slower)
- Increase `rank` to 32 or 64
- Increase `num_iters` to 20000+
- Decrease `learning_rate` to 1e-4
- Increase `batch_size` × `grad_accumulation_steps` to 32

### For Faster Training
- Decrease `rank` to 8
- Use `preset: attention-qv` (fewer parameters)
- Increase `batch_size` if memory allows
- Decrease `num_iters` if dataset is large

### For Limited VRAM
- Decrease `batch_size` to 2
- Increase `grad_accumulation_steps` to 8
- Decrease `max_seq_length` to 1024
- Use `preset: attention-qv` instead of `attention-all`

## Dataset Size Recommendations

| Dataset Size | Recommended Iterations | Training Time (M2 Ultra) |
|--------------|----------------------|--------------------------|
| 1K samples | 1,000 - 2,000 | ~10 minutes |
| 10K samples | 3,000 - 5,000 | ~1 hour |
| 50K samples | 5,000 - 10,000 | ~3 hours |
| 100K+ samples | 10,000 - 20,000 | ~6 hours |

Adjust based on validation loss convergence.

## Troubleshooting

**Out of memory errors:**
- Reduce `batch_size` to 2 or 1
- Reduce `max_seq_length` to 1024
- Use `preset: attention-qv` (fewer parameters)

**Loss not decreasing:**
- Increase `learning_rate` to 5e-4
- Check data quality (validation errors)
- Increase `num_iters`

**Training too slow:**
- Ensure `eager: false` (compiled mode)
- Increase `batch_size` if memory allows
- Use smaller model (Qwen3-0.6B vs 8B)

**Dataset format errors:**
- Check JSONL format: `head -1 data/alpaca/train.jsonl | jq`
- Verify chat format: `{"messages": [{"role": "user", ...}]}`
- Run `lmforge prepare` to validate format
