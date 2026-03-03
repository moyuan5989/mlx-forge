# LMForge — Implementation Guide

> **Status**: V1 Complete (260 tests passing)
> **Full V1 implementation history**: See `CLAUDE_V1_ARCHIVE.md`

---

## 1. Project Overview

**LMForge** is a LoRA SFT training framework for MLX on Apple Silicon with a browser-based UI.

### Current Capabilities

**Training**:
- LoRA + QLoRA fine-tuning (67% memory reduction with 4-bit quantization)
- Sequence packing (2-5x speedup on short sequences)
- Gradient checkpointing (40-60% activation memory savings)
- Resume from checkpoint
- Compiled training loop with gradient accumulation

**Inference**:
- Text generation (greedy, temperature, top-p, repetition penalty)
- Streaming token generation
- Interactive chat REPL
- KV cache for efficient autoregressive decoding

**Studio UI**:
- Browser-based training dashboard (React + FastAPI)
- Real-time metrics via WebSocket
- Loss curve visualization with Recharts
- Model/dataset management
- Interactive playground for testing fine-tuned models

**Supported Architectures**:
- Llama 2/3 (all sizes)
- Mistral (remapped to Llama)
- Qwen3 (0.6B, 1.7B, 4B, 8B)
- Phi-3 (all sizes)
- Gemma 1/2/3 (1B-27B with soft-capping, sliding window)

**CLI Commands**:
```bash
lmforge prepare   --data FILE --model MODEL [--output DIR]
lmforge train     --config FILE [--resume CHECKPOINT_DIR]
lmforge generate  --model MODEL [--adapter DIR] [--prompt TEXT] [...]
lmforge studio    [--host HOST] [--port PORT]
lmforge data      list | catalog | download | import | inspect | stats | delete
```

---

## 2. Core Design Principles

### Frozen Contracts (v0 — DO NOT BREAK)

These contracts are **immutable** and must be preserved in all future versions:

1. **Checkpoint format**: Exactly 3 files per checkpoint
   - `adapters.safetensors` (LoRA weights)
   - `optimizer.safetensors` (optimizer state)
   - `state.json` (step, epoch, trained_tokens, best_val_loss, learning_rate, rng_seed, schema_version)

2. **Batch contract (V2)**: `(B, T)` input_ids + `(B, T)` labels with `-100` masking
   - Standard batches: `iterate_batches()` yields `(input_ids, labels)` — both `(B, T)` int32
   - Packed batches: `iterate_packed_batches()` yields `(input_ids, labels, segment_ids)` — all `(B, T)` int32
   - Preference batches: `iterate_preference_batches()` yields `(chosen_ids, chosen_labels, rejected_ids, rejected_labels)`
   - Labels use `-100` for tokens excluded from loss (prompt tokens, padding, segment boundaries)

3. **Config schema**: Pydantic v2 models with `extra="forbid"`
   - New fields must be optional with backward-compatible defaults
   - Schema version in state.json remains 1

4. **Run directory layout**:
   ```
   ~/.lmforge/runs/{run_id}/
   ├── config.yaml
   ├── manifest.json
   ├── environment.json
   ├── checkpoints/
   │   ├── step-NNNNNNN/
   │   └── best -> step-NNNNNNN
   └── logs/
       └── metrics.jsonl
   ```

5. **Data storage format (V2)**:
   ```
   ~/.lmforge/datasets/
   ├── raw/{dataset_id}/           # Downloaded/imported datasets
   │   ├── data.jsonl
   │   └── meta.json
   └── processed/{name}--{model}/  # Tokenized datasets (Arrow/datasets lib)
       ├── dataset_info.json
       ├── state.json
       ├── data-00000-of-00001.arrow
       └── meta.json
   ```

### Design Philosophy

- **Library-first**: All operations are Python functions; CLI is a thin wrapper
- **Explicit targeting**: LoRA adapters applied via glob patterns (e.g., `*.self_attn.q_proj`)
- **Tier-1 checkpointing**: Save only essential state (not RNG state, data iterator position)
- **Stateless LR schedules**: Pure functions of step number (reconstructed on resume)
- **Fail fast**: Validate all configs before loading models or data
- **No database**: Filesystem is the database (Studio reads `~/.lmforge/` directly)

---

## 3. Architecture & Code Organization

### Package Structure
```
lmforge/
├── adapters/           # LoRA targeting, application, fusing
├── cli/                # 5 commands (prepare, train, generate, studio, data)
├── config.py           # Pydantic config models
├── data/               # Formats, preprocessing, backend, batching, packing, catalog, registry
├── inference/          # Generation engine, sampling, KV cache
├── models/             # Registry, loader, quantization
│   ├── _base/          # Shared utilities (attention, RoPE, activations)
│   └── architectures/  # Llama, Qwen3, Phi-3, Gemma
├── studio/             # FastAPI backend + React frontend (built)
│   ├── api/            # REST endpoints
│   ├── services/       # Run/model/dataset/training services
│   └── frontend/       # Pre-built React SPA
└── trainer/            # Training loop, checkpointing, callbacks
```

### Model Architecture Interface

All architectures follow this interface:

```python
class Model(nn.Module):
    def __call__(self, inputs: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        """
        Args:
            inputs: (B, T) token IDs
            cache: Optional KV cache (one per layer) for inference

        Returns:
            logits: (B, T, vocab_size)
        """
```

Cache interface:
```python
class KVCache:
    offset: int  # Current sequence length

    def update_and_fetch(self, keys, values) -> tuple[mx.array, mx.array]:
        """Concatenate new keys/values, update offset, return full cache."""
```

---

## 4. Key Technical Learnings

### MLX Gotchas

1. **Fancy index assignment limitations**:
   ```python
   # ⚠️ May not work: arr[indices] = values
   # ✅ Workaround: Use boolean mask + mx.where()
   mask_np = np.zeros(arr.shape[0], dtype=bool)
   mask_np[indices] = True
   mask = mx.array(mask_np)
   result = mx.where(mask, new_values, old_values)
   ```

2. **Inverse permutation for scatter**:
   ```python
   sorted_indices = mx.argsort(values)
   inverse = mx.argsort(sorted_indices)  # Inverse permutation
   result = sorted_values[inverse]  # Scatter via gather
   ```

3. **`.at[]` accessor exists**:
   ```python
   # ✅ MLX arrays DO have .at[] accessor (like JAX)
   arr.at[0].add(10.0)
   arr.at[[0, 2]].add(mx.array([10.0, 20.0]))
   ```

4. **QLoRA gradient flow**:
   ```python
   # ✅ Use nn.value_and_grad (only trainable params)
   loss_and_grad = nn.value_and_grad(model, loss_fn)

   # ❌ Don't use mx.value_and_grad (tries to diff through QuantizedMatmul)
   ```

5. **QLoRA freeze order**:
   ```python
   model.freeze()                    # 1. Freeze base model
   nn.quantize(model, bits=4)        # 2. Quantize frozen weights
   apply_lora(model, targets, cfg)   # 3. Apply LoRA (creates unfrozen params)
   ```

6. **API deprecations**:
   ```python
   mx.set_wired_limit()       # ✅ (was mx.metal.set_wired_limit)
   mx.get_peak_memory()       # ✅ (was mx.metal.get_peak_memory)
   ```

### Training Loop Best Practices

```python
# Compiled step function
compile_state = [model.state, optimizer.state, mx.random.state]

@partial(mx.compile, inputs=compile_state, outputs=compile_state)
def step(batch, lengths, prev_grad, do_update):
    (loss, ntoks), grad = loss_and_grad(model, batch, lengths)
    if prev_grad is not None:
        grad = tree_map(lambda a, b: a + b, grad, prev_grad)
    if do_update:
        if grad_accum_steps > 1:
            grad = tree_map(lambda g: g / grad_accum_steps, grad)
        if max_grad_norm is not None:
            grad = clip_grad_norm(grad, max_grad_norm)
        optimizer.update(model, grad)
        grad = None
    return loss, ntoks, grad

# Main loop
for it in range(start_step, num_iters + 1):
    batch, lengths = next(batch_iter)
    do_update = (it % grad_accumulation_steps == 0)

    loss, toks, grad_accum = step(batch, lengths, grad_accum, do_update)
    mx.eval(compile_state, loss, toks, grad_accum)  # SAFE POINT
```

### Frontend Notes

- **Studio frontend**: `studio-frontend/` (React + Vite + TypeScript)
- **Built output**: `lmforge/studio/frontend/` (served by FastAPI)
- **Tailwind CSS v4**: Uses `@import "tailwindcss"` in CSS (no config file)
- **Vite gotcha**: `erasableSyntaxOnly` requires explicit field declarations (no `public` constructor params)
- **Backend**: `RunService._sanitize_for_json()` handles inf/nan floats

---

## 5. Testing

### Run Tests
```bash
.venv/bin/python -m pytest tests/ -v
```

### Test Coverage (260 tests)
- **48** core tests (v0: M1-M6)
- **14** M7 tests (HuggingFace model loading)
- **20** M8 tests (self-contained model loading)
- **40** M9 tests (resume, inference, Gemma)
- **37** M10 tests (QLoRA, gradient checkpointing, packing)
- **50** M11 tests (Studio backend)
- **25** M12 tests (Studio frontend)
- **26** M13 tests (integration)

### Test Files
- `test_config.py` — Config validation
- `test_data.py` — Data pipeline (batching, fingerprinting)
- `test_adapters.py` — LoRA targeting and application
- `test_model_loading.py` — Model loading and architecture
- `test_resolve.py` — HF model resolution
- `test_trainer_infra.py` — Trainer components
- `test_integration.py` — End-to-end workflows
- `test_m9_foundation.py` — Resume, inference, Gemma
- `test_m10_performance.py` — QLoRA, packing, gradient checkpointing
- `test_m11_studio.py` — Studio backend
- `test_m12_frontend.py` — Studio frontend
- `test_m13_integration.py` — Cross-feature integration
- `test_labels.py` — Per-token label construction (chat, completions, text, preference)
- `test_backend.py` — Arrow storage backend (save/load, metadata, fingerprints)
- `test_catalog.py` — Dataset catalog, converters, registry
- `test_v2_dpo.py` — DPO/preference training
- `test_v2_studio.py` — V2 contracts and architectures

---

## 6. Configuration Reference

### Training Config Structure
```yaml
schema_version: 1

model:
  path: "Qwen/Qwen3-0.6B"           # HF ID or local path
  revision: "abc123"                 # Optional: pin to specific commit
  quantization:                      # Optional: QLoRA
    bits: 4                          # 4 or 8
    group_size: 64                   # 32, 64, or 128

adapter:
  preset: "attention-qv"             # Or: attention-all, mlp, all-linear
  # targets: ["*.q_proj", "*.v_proj"]  # Alternative: custom patterns
  rank: 16
  scale: 32.0
  num_layers: 16                     # Optional: apply to last N layers only

data:
  train: "./train.jsonl"
  valid: "./val.jsonl"
  packing: false                     # Optional: sequence packing
  max_seq_length: 2048

training:
  optimizer: "adam"
  learning_rate: 1.0e-5
  num_iters: 1000
  batch_size: 4
  grad_accumulation_steps: 1
  gradient_checkpointing: false      # Optional: activation checkpointing
  steps_per_save: 100
  steps_per_eval: 200
  steps_per_report: 10
  max_grad_norm: 1.0

runtime:
  run_dir: "~/.lmforge/runs"
  seed: 42
  eager: false                       # true = disable mx.compile
```

### Data Formats (JSONL)

**Chat format**:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Completions format**:
```json
{"prompt": "...", "completion": "..."}
```

**Text format**:
```json
{"text": "..."}
```

**Preference format** (DPO):
```json
{"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

## 7. Next Steps & Future Work

### Immediate Priorities
- [ ] Update README with V2 features
- [ ] Add Studio user guide
- [ ] Create example configs for QLoRA, packing, gradient checkpointing
- [ ] Document Phi-3 OOM findings and memory optimization

### Performance Benchmarking
- [ ] Measure QLoRA memory savings (expected: 67%)
- [ ] Measure packing speedup on Alpaca (expected: 2-5x)
- [ ] Profile gradient checkpointing overhead (expected: ~30% compute)
- [ ] Compare throughput across architectures

### V2 Feature Candidates
- [ ] DoRA adapters (Magnitude-preserving LoRA)
- [ ] Additional architectures:
  - DeepSeek-R1 / DeepSeek-V3 (if MoE support added)
- [ ] Multi-dataset mixing with sampling strategies
- [ ] Evaluation harness integration (lm-eval)
- [ ] Native desktop app (Tauri wrapper for Studio)

### Known Limitations (By Design)
V1/V2 intentionally does NOT implement:
- Inference serving / REST API / OpenAI compatibility
- Distributed training (Apple Silicon is single-GPU)
- MoE models (Mixtral, DeepSeek V3)
- Model conversion (HF ↔ MLX)
- Multi-dataset mixing
- Cloud sync / multi-user collaboration

---

## 8. Git Workflow & Commits

### Commit Message Format
```
<type>: <subject>

<optional body>

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

### Never Commit
- `.env` files
- Credentials
- Large binaries (models, datasets)
- `__pycache__/`, `*.pyc`
- `.venv/`, `node_modules/`
- Downloaded datasets (add to `.gitignore`)

---

## 9. Dependencies

### Core (Required)
```
mlx>=0.18.0
pydantic>=2.0
pyyaml>=6.0
numpy>=1.24.0
transformers>=4.35.0
safetensors>=0.4.0
huggingface-hub>=0.20.0
datasets>=2.16.0
```

### Optional
```
# Studio UI
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0

# Logging
wandb>=0.16.0

# Development
pytest>=7.0
pytest-timeout>=2.0
pytest-asyncio>=0.23.0
httpx>=0.25.0
```

Install: `pip install lmforge[studio,wandb,dev]`

---

## 10. Quick Reference

### Add New Architecture

1. **Create architecture file**: `lmforge/models/architectures/newmodel.py`
2. **Implement interface**:
   ```python
   class NewModelArgs(BaseModelArgs):
       @classmethod
       def from_dict(cls, config: dict) -> "NewModelArgs": ...

   class NewModel(nn.Module):
       def __call__(self, inputs, cache=None) -> mx.array: ...
   ```
3. **Register**: Add to `lmforge/models/registry.py`
4. **Test**: Add to `tests/test_model_loading.py`

### Add New LoRA Preset

Edit `lmforge/adapters/targeting.py`:
```python
PRESETS = {
    "my-preset": ["*.module1", "*.module2"],
}
```

### Add New LR Schedule

Edit `lmforge/trainer/optimizer.py`:
```python
def build_scheduler(config: TrainingParams):
    if config.lr_schedule.name == "my_schedule":
        return my_schedule_fn(config.learning_rate, ...)
```

---

## 11. Troubleshooting

### Out of Memory (OOM)
1. Enable QLoRA: `model.quantization.bits: 4`
2. Enable gradient checkpointing: `training.gradient_checkpointing: true`
3. Reduce batch size: `training.batch_size: 1-2`
4. Increase grad accumulation: `training.grad_accumulation_steps: 8`
5. Reduce sequence length: `data.max_seq_length: 1024`

### Training Loss Not Decreasing
1. Check learning rate (try 1e-5 to 1e-4 for LoRA)
2. Verify adapter targeting (check console output for "Applied LoRA to N modules")
3. Check data quality (validate JSONL samples)
4. Ensure `mask_prompt: true` if training on completions only

### Resume Not Working
1. Verify checkpoint has 3 files: `adapters.safetensors`, `optimizer.safetensors`, `state.json`
2. Check `num_iters` > checkpoint step
3. Use full path to checkpoint directory (not parent)

### Studio Not Starting
1. Install Studio dependencies: `pip install lmforge[studio]`
2. Check port availability: `lsof -i :8741`
3. Try different port: `lmforge studio --port 8742`

---

**Full V1 Implementation History**: See `CLAUDE_V1_ARCHIVE.md`
