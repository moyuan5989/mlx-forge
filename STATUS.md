# LMForge v0 — Project Status

> Last updated: 2026-02-01

---

## Summary

**M0: Scaffolding ✅ COMPLETE**
**M1: Config System ✅ COMPLETE**
**M2: Data Pipeline ✅ COMPLETE**
**M3: Model + Adapters ✅ COMPLETE**
**M4: Trainer Infrastructure ✅ COMPLETE**
**M5: Trainer + Run Management ✅ COMPLETE**

Full end-to-end training system with compiled training loop, run orchestration, and manifest generation. 45 passing tests (14 M1 + 14 M2 + 10 M3 + 7 M4), ready for integration testing.

---

## What's Been Accomplished

### 1. Documentation & Planning

- ✅ **IMPLEMENTATION_PLAN.md** — 7 milestones (M0–M6) mapping to CLAUDE.md implementation steps
- ✅ **V0_DESIGN_FREEZE.md** — Frozen contracts (config schema, batch format, checkpoint layout)
- ✅ **CLAUDE.md** — Authoritative implementation guide with ground rules

### 2. Root Configuration

- ✅ **pyproject.toml** — Complete build config with:
  - Dependencies: `mlx>=0.18.0`, `pydantic>=2.0`, `pyyaml`, `numpy`, `transformers`, `safetensors`
  - Dev dependencies: `pytest>=7.0`, `pytest-timeout>=2.0`
  - Optional dependencies: `wandb` (for WandB integration)
  - Console script: `lmforge = lmforge.cli.main:main`
- ✅ **.gitignore** — Standard Python excludes + `.lmforge/` runtime dir

### 3. Package Structure (22 Python Files)

#### Core (`lmforge/`)
- ✅ `__init__.py` — Public API with `prepare()` and `train()` stubs
- ✅ `_version.py` — `__version__ = "0.1.0"`
- ✅ **`config.py` — FULLY IMPLEMENTED**
  - All 7 Pydantic models complete with validators
  - `TrainingConfig.from_yaml()` working
  - `extra="forbid"` on all models
  - Mutual exclusion validator for `targets`/`preset`
  - `steps_per_save % grad_accumulation_steps` validator
- ✅ `manifest.py` — `RunManifest`, `EnvironmentInfo` dataclasses (stubs)

#### Data Pipeline (`lmforge/data/`)
- ✅ `formats.py` — Format detection + validation (stub)
- ✅ `preprocessing.py` — Tokenization + offset computation (stub)
- ✅ `cache.py` — Fingerprinting + safetensors I/O (stub)
- ✅ `batching.py` — Sort-by-length + fixed-batch iterator (stub)

#### Adapters (`lmforge/adapters/`)
- ✅ `targeting.py` — **PRESETS dict defined**, glob matching (stub)
- ✅ `lora.py` — `LoRALinear`, `LoRAEmbedding`, `apply_lora()` (stubs)

#### Trainer (`lmforge/trainer/`)
- ✅ **`state.py` — FULLY IMPLEMENTED** (`TrainState` dataclass)
- ✅ **`callbacks.py` — Callback + CallbackList FULLY IMPLEMENTED**
  - Base `Callback` class with 5 hooks
  - `CallbackList` dispatcher complete
  - `MetricsLoggerCallback`, `ConsoleCallback`, `WandBCallback` (stubs)
- ✅ `trainer.py` — `Trainer` class (stub)
- ✅ `checkpoint.py` — `CheckpointManager` (stub)
- ✅ `optimizer.py` — `build_optimizer()`, `build_scheduler()` (stubs)

#### Models (`lmforge/models/`)
- ✅ `loader.py` — `load_model()` (stub)

#### Logging (`lmforge/logging/`)
- ✅ `metrics.py` — JSONL writer + console formatter (stubs)

#### CLI (`lmforge/cli/`)
- ✅ **`main.py` — FULLY IMPLEMENTED**
  - Complete argparse with `prepare` and `train` subcommands
  - `--help`, `--version` working
  - Delegates to separate command handlers
- ✅ `prepare_cmd.py` — `run_prepare()` (stub)
- ✅ `train_cmd.py` — `run_train()` (stub)

### 4. Examples

- ✅ **examples/train.yaml** — Valid config matching V0_DESIGN_FREEZE.md §2.1
  - Model: `Qwen/Qwen3-0.6B`
  - Adapter: `preset: attention-qv`, rank 8
  - Training: 1000 iters, batch size 4, Adam optimizer

### 5. Tests (7 Files, 38 Passing + 10 Pending)

- ✅ `conftest.py` — Fixtures for `tmp_dir` and `sample_config_dict`
- ✅ **`test_config.py` — 14 TESTS PASSING (M1 COMPLETE)**
  - Config loading from dict and YAML
  - Schema version validation
  - Extra fields rejection (`extra="forbid"`)
  - Missing required fields
  - Invalid optimizer enum
  - Adapter targets/preset mutual exclusion
  - TrainingParams validation (steps_per_save % grad_accumulation_steps)
  - LRScheduleConfig optional/required
  - RuntimeConfig defaults
- ✅ **`test_data.py` — 14 TESTS PASSING (M2 COMPLETE)**
  - Format detection (chat, completions, text)
  - Format validation with comprehensive error checking
  - Fingerprinting (SHA-256 of data + tokenizer + template)
  - Cache write/read with safetensors shards
  - Cache hit detection
  - Batching with correct shapes (B, T) and (B, 2)
  - Padding to multiple of 32
- ✅ **`test_adapters.py` — 10 TESTS PASSING (M3 COMPLETE)**
  - Preset resolution (attention-qv, attention-all, mlp, all-linear)
  - Glob pattern matching with fnmatch
  - num_layers filtering for last N layers
  - LoRALinear.from_base() and fuse()
  - LoRAEmbedding.from_base() and fuse()
  - apply_lora() integration
  - named_modules() recursive enumeration
- ✅ **`test_trainer_infra.py` — 7 TESTS PASSING (M4 COMPLETE)**
  - Optimizer factory (Adam with LR)
  - LR schedule changes over steps
  - Checkpoint save produces 3 files
  - Load restores state
  - Atomic write uses tmp dir
  - Retention keeps last N + best
  - JSONL output format
- ⏸️ `test_integration.py` — 3 tests for M6 (all skip)

### 6. Verification ✅ All Passing

```bash
# Package installation
pip install -e ".[dev]"  # ✅ SUCCESS

# Imports
python -c "from lmforge import prepare, train; from lmforge.config import TrainingConfig"  # ✅ OK

# CLI
lmforge --help           # ✅ Shows usage
lmforge prepare --help   # ✅ Shows prepare options
lmforge train --help     # ✅ Shows train options

# Tests
pytest tests/ -v         # ✅ 45 passed, 3 skipped

# Config loading
python -c "from lmforge.config import TrainingConfig; c = TrainingConfig.from_yaml('examples/train.yaml')"  # ✅ OK
```

---

## What's Been Accomplished in M2

### Data Pipeline Implementation

✅ **Format Detection** (`data/formats.py`):
- `detect_format()` — auto-detects chat/completions/text from sample keys
- `validate_samples()` — validates all samples, collects all errors before reporting
- Comprehensive validation for all three formats

✅ **Tokenization** (`data/preprocessing.py`):
- `tokenize_dataset()` — applies chat template, computes prompt offsets
- Chat format: re-encodes without last message to compute offset if `mask_prompt=True`
- Completions format: wraps in chat format then processes
- Text format: simple encode, offset=0, appends EOS if missing

✅ **Caching** (`data/cache.py`):
- `compute_fingerprint()` — SHA-256 of (data_hash + tokenizer_hash + template_hash)
- `write_cache()` — writes safetensors shards (~500MB each) + meta.json
- `read_cache()` — loads from cache
- `check_cache()` — cache hit detection with integrity validation
- Shard format matches V0_DESIGN_FREEZE.md §2.5 exactly

✅ **Batching** (`data/batching.py`):
- `iterate_batches()` — sort by length, fixed batch size, pad to multiple of 32
- Returns `(batch_tokens, lengths)` per V0_DESIGN_FREEZE.md §2.2 contract
- Batch contract verified: shapes (B, T) and (B, 2), dtype int32

✅ **CLI + API** (`__init__.py`, `cli/prepare_cmd.py`):
- `lmforge.prepare()` — full implementation with progress reporting
- `lmforge prepare` CLI command working
- Cache hit/miss detection, statistics reporting

### Files Modified (M2)

- `lmforge/data/formats.py` — 135 lines (format detection + validation)
- `lmforge/data/preprocessing.py` — 124 lines (tokenization)
- `lmforge/data/cache.py` — 190 lines (caching + fingerprinting)
- `lmforge/data/batching.py` — 88 lines (batching)
- `lmforge/__init__.py` — 117 lines (`prepare()` implementation)
- `lmforge/cli/prepare_cmd.py` — 23 lines (CLI handler)
- `tests/test_data.py` — 254 lines (14 comprehensive tests)

---

## What's Been Accomplished in M3

### Model Loading + LoRA Adapters

✅ **Model Loader** (`models/loader.py`):
- `load_model()` — loads model + tokenizer via mlx_lm
- Optional mlx_lm dependency (clear error if not installed)
- Trust remote code support

✅ **Adapter Targeting** (`adapters/targeting.py`):
- `named_modules()` — recursive module enumeration for MLX models
- `get_patterns()` — resolve config.targets or config.preset
- `resolve_targets()` — glob matching with fnmatch.fnmatch()
- `num_layers` filtering for last N transformer layers
- PRESETS: attention-qv, attention-all, mlp, all-linear
- Clear error messages with available module paths

✅ **LoRA Implementation** (`adapters/lora.py`):
- `LoRALinear` class with proper initialization (Kaiming for A, zeros for B)
- `LoRAEmbedding` class for embedding table adaptation
- `from_base()` classmethod for creating from existing layers
- `fuse()` method to merge LoRA weights back: W' = W + (scale/r) * B @ A
- `apply_lora()` to apply adapters in-place with tree_unflatten
- Dropout support, base layer freezing
- Trainable parameter counting + logging

### Files Modified (M3)

- `lmforge/models/loader.py` — 41 lines (model loading with mlx_lm)
- `lmforge/adapters/targeting.py` — 152 lines (glob matching + layer filtering)
- `lmforge/adapters/lora.py` — 277 lines (LoRA adapters)
- `tests/test_adapters.py` — 201 lines (10 comprehensive tests)

---

## What's Been Accomplished in M4

### Trainer Infrastructure

✅ **Optimizer Factory** (`trainer/optimizer.py`):
- `build_optimizer()` — creates Adam, AdamW, SGD, or Adafactor optimizers from config
- `build_scheduler()` — builds stateless LR schedules (cosine_decay, linear_schedule, step_decay, exponential_decay)
- Warmup support via `join_schedules()`
- Pure functions of step number for resume compatibility

✅ **Checkpoint Manager** (`trainer/checkpoint.py`):
- `save()` — atomic checkpoint save with exactly 3 files (adapters.safetensors, optimizer.safetensors, state.json)
- `load()` — restore model, optimizer, and training state
- Retention policy: keep last N checkpoints + best
- Best symlink management
- Tmp-dir-then-rename pattern for atomic writes
- Schema version 1 support with forward compatibility check

✅ **Callbacks** (`trainer/callbacks.py`):
- `MetricsLoggerCallback` — writes JSONL metrics to logs/metrics.jsonl
- `ConsoleCallback` — prints human-readable progress to stdout
- `WandBCallback` — optional W&B integration with try/except import
- Base `Callback` and `CallbackList` classes (from M0) used correctly

✅ **Metrics Logging** (`logging/metrics.py`):
- `write_metrics_line()` — appends JSONL line with auto-added timestamp
- `format_console_line()` — formats metrics for console display
- Two event types: "train" and "eval"
- ISO 8601 timestamps with Z suffix

### Files Modified (M4)

- `lmforge/trainer/optimizer.py` — 124 lines (optimizer + scheduler factory)
- `lmforge/trainer/checkpoint.py` — 188 lines (checkpoint manager)
- `lmforge/trainer/callbacks.py` — 104 lines (3 callback implementations)
- `lmforge/logging/metrics.py` — 53 lines (JSONL writer + console formatter)
- `tests/test_trainer_infra.py` — 287 lines (7 comprehensive tests)

---

## What's Been Accomplished in M5

### Trainer + Run Management

✅ **Trainer Class** (`trainer/trainer.py`):
- `Trainer.__init__()` — initializes with model, config, datasets, callbacks, optimizer, checkpoint manager
- `loss_fn()` — cross-entropy loss with prompt masking per V0_DESIGN_FREEZE.md §2.2
- `loss_value_and_grad()` — computes loss and gradients
- `clip_grad_norm()` — gradient clipping by global norm
- Compiled step function with gradient accumulation support
- `fit()` — full training loop with:
  - Metal memory limit configuration
  - Callback boundaries (on_train_begin, on_step_end, on_eval_end, on_save, on_train_end)
  - Evaluation at step 1, every steps_per_eval, and final step
  - Metrics reporting every steps_per_report
  - Checkpoint saves every steps_per_save and at final step
  - Cooperative pause support (optional feature)
  - Safe points with mx.eval() for callback execution
- `evaluate()` — validation loop that computes average loss over validation set

✅ **Manifest & Environment** (`manifest.py`):
- `collect_environment()` — collects Python version, MLX version, lmforge version, platform, OS, chip, memory, GPU cores
- `write_manifest()` — writes manifest.json and environment.json to run directory
- HardwareInfo collection via mx.metal.device_info() on Apple Silicon
- ISO 8601 timestamps with Z suffix

✅ **Top-level train() Function** (`__init__.py`):
- Config loading from YAML or TrainingConfig object
- Run directory creation with unique run ID
- config.yaml writing to run directory
- Model and tokenizer loading
- LoRA adapter application with progress reporting
- Automatic data preparation (cache hit/miss detection)
- Training and validation dataset loading
- Manifest and environment.json generation
- Callback creation (Console, MetricsLogger, optional WandB)
- Trainer initialization and execution
- Final summary with best val loss and checkpoint location

✅ **CLI Command** (`cli/train_cmd.py`):
- Already complete from M0 — delegates to train() function

### Files Modified (M5)

- `lmforge/trainer/trainer.py` — 258 lines (full Trainer implementation)
- `lmforge/manifest.py` — 140 lines (environment collection + manifest writing)
- `lmforge/__init__.py` — 225 lines (train() function)
- `lmforge/cli/train_cmd.py` — 12 lines (already complete)

---

## What's Next: M6 — Integration Testing

**Target**: End-to-end integration tests and final verification.

### Deliverables

1. **Format detection** (`data/formats.py`):
   - `detect_format()` — auto-detect chat/completions/text from sample keys
   - `validate_samples()` — validate all samples match schema, collect errors

2. **Tokenization** (`data/preprocessing.py`):
   - `tokenize_dataset()` — apply chat template, compute offsets
   - Handle chat, completions, and text formats
   - Support `mask_prompt` flag for loss masking

3. **Caching** (`data/cache.py`):
   - `compute_fingerprint()` — SHA-256 of data + tokenizer + template
   - `write_cache()` — save safetensors shards (~500MB each) + meta.json
   - `read_cache()` — load from cache
   - `check_cache()` — cache hit detection

4. **Batching** (`data/batching.py`):
   - `iterate_batches()` — sort by length, fixed batch size, pad to multiple of 32
   - Return `(batch_tokens, lengths)` per V0_DESIGN_FREEZE.md §2.2

5. **CLI command** (`cli/prepare_cmd.py`):
   - `run_prepare()` — full implementation
   - Calls `lmforge.prepare()` with CLI args

6. **Library API** (`lmforge.__init__.py`):
   - `prepare()` — full implementation

### Tests to Implement

- `test_detect_chat_format`, `test_detect_completions_format`, `test_detect_text_format`
- `test_unknown_format_raises`
- `test_same_inputs_same_fingerprint`, `test_different_data_different_fingerprint`
- `test_batch_shapes_match_contract`, `test_padding_to_multiple_of_32`

### Done When

- All 8 `test_data.py` tests pass
- `lmforge prepare` runs end-to-end
- Cache shards match V0_DESIGN_FREEZE.md §2.5 layout
- Re-running skips tokenization (cache hit)
- Batch shapes match contract: `(B, T)` and `(B, 2)`

---

## Full Roadmap Ahead

| Milestone | Status | Key Deliverables |
|-----------|--------|------------------|
| **M0: Scaffolding** | ✅ **COMPLETE** | 39 files, package installable, all verifications pass |
| **M1: Config System** | ✅ **COMPLETE** | 14 tests passing, all validators working |
| **M2: Data Pipeline** | ✅ **COMPLETE** | 14 tests passing, `lmforge prepare` working |
| **M3: Model + Adapters** | ✅ **COMPLETE** | 10 tests passing, LoRA + targeting complete |
| **M4: Trainer Infra** | ✅ **COMPLETE** | 7 tests passing, optimizer, checkpoints, callbacks, metrics |
| **M5: Trainer + Run** | ✅ **COMPLETE** | Full training loop, run management, manifest generation |
| **M6: Integration** | 🎯 **NEXT** | End-to-end tests, resume validation |

---

## Key Design Principles

1. **Library-first** — All operations are Python functions; CLI is a thin wrapper
2. **Frozen contracts** — Config schema, batch format, checkpoint layout are immutable
3. **Explicit adapter targeting** — Glob patterns on module paths, no type-based scanning
4. **Tier-1 checkpointing** — Adapters + optimizer + state only (no RNG state, no data iterator position)
5. **Stateless LR schedules** — Pure functions of step number, no saved scheduler state
6. **Fail fast** — Validate all configs before loading models or data

---

## Quick Start (Once M2+ Are Done)

```bash
# Prepare data
lmforge prepare --data train.jsonl --model meta-llama/Llama-3.2-3B-Instruct

# Train
lmforge train --config examples/train.yaml

# Resume
lmforge train --config examples/train.yaml --resume ~/.lmforge/runs/{run_id}/checkpoints/step-0001000
```

---

## Notes

- **Virtual environment**: Created at `.venv/` (excluded from git)
- **MLX version**: 0.30.4 (installed)
- **Python version**: 3.14.2 (user's system)
- **No mlx-lm dependency**: Decision deferred to M3 (model loader implementation)
