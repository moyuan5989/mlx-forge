# LMForge v0 — Implementation Command File

> This file is the authoritative implementation guide for LMForge v0.
> All contracts reference V0_DESIGN_FREEZE.md. Do not deviate.

---

## 1. Mission Statement

Build LMForge v0: a LoRA SFT training framework for MLX on Apple Silicon.

v0 ships exactly two CLI commands (`lmforge prepare` and `lmforge train`), a Python library API, and structured run artifacts. Nothing else.

---

## 2. Ground Rules

1. **Library-first**: Every operation is a Python function with typed arguments. The CLI is a thin wrapper that parses YAML/args into Pydantic config objects and calls library functions. No business logic in CLI code.

2. **Frozen contracts**: The config schema, checkpoint format, batch contract, and run directory layout defined in `V0_DESIGN_FREEZE.md` are immutable. Do not modify field names, types, or directory structures.

3. **Explicit adapter targeting**: Adapters are applied via glob patterns on module paths using `fnmatch.fnmatch()`. No type-based scanning. No implicit "apply to all linear layers" behavior.

4. **Tier-1 checkpointing only**: Each checkpoint contains exactly three files: `adapters.safetensors` (adapter weights), `optimizer.safetensors` (optimizer moments), and `state.json` (step, epoch, trained_tokens, best_val_loss, learning_rate, rng_seed, schema_version). Do not save `mx.random.state`, numpy random state, data iterator position, or gradient accumulation buffers. See V0_DESIGN_FREEZE.md §2.3.

5. **Stateless LR schedules**: LR schedules are pure functions of step number. On resume, the scheduler is reconstructed from config and given the saved step. No scheduler internal state is saved.

6. **No scope creep**: Do not implement inference, serving, DPO, packing, dynamic batching, distributed training, DoRA, gradient checkpointing, or any feature listed in the out-of-scope section of `V0_DESIGN_FREEZE.md`.

7. **Fail fast**: Validate all configs before loading models or data. Every error message must state what was wrong and what was expected.

8. **No unnecessary dependencies**: Core dependencies are `mlx`, `pydantic>=2.0`, `pyyaml`, `numpy`, `transformers` (for tokenizer loading), `safetensors`. Do not add `pyarrow`, `datasets`, `wandb`, `fastapi`, or any other package as a hard dependency. Optional integrations (WandB) use try/except imports.

---

## 3. v0 Deliverables

### `lmforge prepare`

Pre-tokenizes a dataset and writes a safetensors cache to disk.

```bash
lmforge prepare --data ./train.jsonl --model meta-llama/Llama-3.2-3B-Instruct --output ~/.lmforge/cache/preprocessed/
```

Behavior:
- Loads tokenizer from the model path.
- Reads the JSONL file into memory.
- Auto-detects format (chat / completions / text).
- Validates all samples.
- Tokenizes with chat template and computes prompt offsets.
- Writes safetensors shards + `meta.json`.
- Prints statistics: sample count, total tokens, min/mean/max length, format detected.

Library API:
```python
from lmforge import prepare
stats = prepare(data_path="./train.jsonl", model="meta-llama/Llama-3.2-3B-Instruct")
```

### `lmforge train`

Runs LoRA SFT training from a config file.

```bash
lmforge train --config train.yaml
```

Behavior:
- Loads and validates config.
- Creates run directory.
- Writes `config.yaml`, `manifest.json`, `environment.json`.
- Loads model and tokenizer.
- Applies LoRA adapters per targeting config.
- Loads preprocessed data from cache (or runs prepare automatically if cache miss).
- Runs training loop with compiled step.
- Saves checkpoints per schedule.
- Logs metrics to `metrics.jsonl`.
- Prints summary on completion.

Library API:
```python
from lmforge import train
result = train(config="train.yaml")
# or
from lmforge.config import TrainingConfig
config = TrainingConfig.from_yaml("train.yaml")
result = train(config=config)
```

### Run Artifacts

Every `lmforge train` invocation produces (see V0_DESIGN_FREEZE.md §2.4):
```
~/.lmforge/runs/{run_id}/
├── config.yaml
├── manifest.json
├── environment.json
├── checkpoints/
│   ├── step-NNNNNNN/
│   │   ├── adapters.safetensors
│   │   ├── optimizer.safetensors
│   │   └── state.json
│   └── best -> step-NNNNNNN
└── logs/
    └── metrics.jsonl
```

### Dataset Cache

```
~/.lmforge/cache/preprocessed/{data_fingerprint}/
├── meta.json
├── shard_000.safetensors
└── shard_001.safetensors
```

---

## 4. Python Package Layout

```
lmforge/
├── __init__.py                 # Public API: prepare(), train()
├── _version.py                 # __version__ = "0.1.0"
├── config.py                   # All Pydantic config models
├── manifest.py                 # RunManifest, EnvironmentInfo dataclasses
│
├── data/
│   ├── __init__.py
│   ├── formats.py              # Format detection, schema validation
│   ├── preprocessing.py        # Tokenization, template application, offset computation
│   ├── cache.py                # Safetensors cache write/read, fingerprinting
│   └── batching.py             # Sort-by-length, fixed-batch, pad-to-32 iterator
│
├── adapters/
│   ├── __init__.py
│   ├── targeting.py            # Glob matching, preset resolution, module discovery
│   └── lora.py                 # LoRALinear, LoRAEmbedding, from_base(), fuse()
│
├── trainer/
│   ├── __init__.py
│   ├── trainer.py              # Trainer class: fit(), evaluate()
│   ├── state.py                # TrainState dataclass
│   ├── callbacks.py            # Callback base class + MetricsLogger + WandBCallback
│   ├── checkpoint.py           # CheckpointManager: save, load, retain, atomic write
│   └── optimizer.py            # Optimizer + scheduler factory from config
│
├── models/
│   ├── __init__.py
│   └── loader.py               # Download, load model + tokenizer, apply adapters
│
├── logging/
│   ├── __init__.py
│   └── metrics.py              # JSONL metrics writer, console reporter
│
└── cli/
    ├── __init__.py
    ├── main.py                 # CLI entry point (click or argparse)
    ├── prepare_cmd.py          # lmforge prepare
    └── train_cmd.py            # lmforge train
```

Do not create modules for: inference, generation, serving, recipes, runtime, daemon, export, profiling.

---

## 5. Config Schema Specification

All configs are Pydantic v2 `BaseModel` subclasses. Use `model_config = ConfigDict(extra="forbid")` to reject unknown fields. The full schema definitions are frozen in V0_DESIGN_FREEZE.md §2.1. This section adds implementation details.

### Validators

#### AdapterConfig validator

```python
@model_validator(mode="after")
def validate_targeting(self) -> "AdapterConfig":
    if self.targets is not None and self.preset is not None:
        raise ValueError("Specify 'targets' or 'preset', not both.")
    if self.targets is None and self.preset is None:
        raise ValueError(
            "Must specify 'targets' (glob patterns) or 'preset'. "
            "Available presets: attention-qv, attention-all, mlp, all-linear."
        )
    return self
```

#### TrainingParams validator

```python
@model_validator(mode="after")
def validate_save_accum(self) -> "TrainingParams":
    if self.steps_per_save % self.grad_accumulation_steps != 0:
        raise ValueError(
            f"steps_per_save ({self.steps_per_save}) must be a multiple of "
            f"grad_accumulation_steps ({self.grad_accumulation_steps})."
        )
    return self
```

### Config Loading

```python
class TrainingConfig(BaseModel):
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
```

CLI args can override individual fields. Config file is the primary source. CLI overrides are applied as dict updates before Pydantic validation.

---

## 6. Manifest vs Config

### Separation

- **Config** (`TrainingConfig`): What the user wants. Mutable during setup. Validated once. Input to the system.
- **Manifest** (`RunManifest`): What actually happened. Frozen at run start. Saved with artifacts. Output of the system.

### RunManifest

```python
@dataclass
class RunManifest:
    schema_version: int
    config: dict                        # frozen TrainingConfig as dict
    lmforge_version: str
    mlx_version: str
    python_version: str
    hardware: HardwareInfo
    data_fingerprint: str               # from preprocessed cache meta.json
    created_at: str                     # ISO 8601

@dataclass
class HardwareInfo:
    chip: str                           # e.g., "Apple M2 Ultra"
    memory_gb: int
    gpu_cores: int
    os: str                             # e.g., "Darwin 24.6.0"
```

### EnvironmentInfo

Saved separately as `environment.json` for quick inspection without parsing the full manifest.

```python
@dataclass
class EnvironmentInfo:
    python_version: str
    mlx_version: str
    lmforge_version: str
    platform: str
    os_version: str
    chip: str
    memory_gb: int
    gpu_cores: int
```

Collect hardware info via:
- `platform.machine()`, `platform.platform()`
- `mx.metal.device_info()` if `mx.metal.is_available()`
- `os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")` for memory

---

## 7. Data Pipeline Implementation Rules

### Prepare Step

1. Load tokenizer: `AutoTokenizer.from_pretrained(model_path, trust_remote_code=...)`.
2. Read JSONL: `[json.loads(line) for line in open(path)]`.
3. Detect format: inspect keys of `data[0]`.
   - Has `"messages"` → chat format.
   - Has `"prompt"` and `"completion"` → completions format.
   - Has `"text"` → text format.
   - Otherwise → raise error listing found keys.
4. Validate: iterate all samples, check they match detected schema. Collect errors, report all at once.
5. Tokenize:
   - Chat: `tokenizer.apply_chat_template(messages, return_dict=False)`. If `mask_prompt=True`, compute offset by re-encoding without the last message. If `mask_prompt=False`, set offset = 0 (loss computed on all tokens including prompt).
   - Completions: wrap in chat format `[{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]`, then process as chat.
   - Text: `tokenizer.encode(text)`. Append EOS if not present. Offset = 0 (regardless of `mask_prompt` setting).
6. Compute fingerprint: `sha256(data_hash + tokenizer_hash + template_hash)` (see V0_DESIGN_FREEZE.md §2.5).
7. Write shards: group samples into ~500MB shards. Each shard is a safetensors file with keys `tokens_{i}` (int32 array) and combined `offsets` and `lengths` arrays (both int32).
8. Write `meta.json` with statistics (see V0_DESIGN_FREEZE.md §2.5 for schema).

### Cache Format

Per V0_DESIGN_FREEZE.md §2.5:

```
shard_NNN.safetensors:
    tokens_0:   int32[L0]      # token IDs for sample 0
    tokens_1:   int32[L1]      # token IDs for sample 1
    ...
    offsets:    int32[N]        # prompt offset per sample in this shard
    lengths:    int32[N]        # total token count per sample in this shard
```

### Fingerprinting

Tokenizer vocab hash: `sha256(json.dumps(sorted(tokenizer.get_vocab().items())))`.
Template hash: `sha256(tokenizer.chat_template or "")`.
Data file hash: `sha256(file_bytes)`.
Combined: `sha256(data_hash + tokenizer_hash + template_hash)`.

### Cache Hit

Before tokenizing, check if `cache_dir/{fingerprint}/meta.json` exists. If yes, load from cache. If no, run prepare.

---

## 8. Adapter Targeting Implementation Rules

### Module Path Enumeration

After loading the model, enumerate all module paths. MLX `nn.Module` does not provide `named_modules()` directly, so implement a recursive helper:

```python
def named_modules(module, prefix=""):
    """Yield (name, module) pairs for all submodules recursively."""
    yield prefix, module
    for name, child in module.children().items():
        full_name = f"{prefix}.{name}" if prefix else name
        yield from named_modules(child, full_name)
```

This produces paths like:
- `model.layers.0.self_attn.q_proj`
- `model.layers.0.self_attn.k_proj`
- `model.layers.0.mlp.gate_proj`
- `model.embed_tokens`

### Glob Matching

Use `fnmatch.fnmatch` on the dot-separated path. Note: `fnmatch` treats `.` as a regular character, so `*` matches across dot boundaries (see V0_DESIGN_FREEZE.md §4 for full semantics).

```python
import fnmatch

def resolve_targets(model, patterns: list[str], num_layers: Optional[int]) -> list[tuple[str, nn.Module]]:
    all_modules = list(named_modules(model))
    matched = []
    for name, module in all_modules:
        if not name:
            continue  # skip root module
        # Apply num_layers filter
        if num_layers is not None:
            if not _is_in_last_n_layers(name, model, num_layers):
                continue
        # Match against any pattern
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                matched.append((name, module))
                break
    if not matched:
        available = [name for name, _ in all_modules if name][:20]
        raise ValueError(
            f"No modules matched patterns {patterns}. "
            f"Available paths (first 20): {available}"
        )
    return matched
```

### `_is_in_last_n_layers`

Extract layer index from path. If path contains `layers.{N}`, check `N >= total_layers - num_layers`. If path does not contain a layer index (e.g., `model.embed_tokens`), skip it (do not apply adapter to non-layer modules when `num_layers` is set).

### Preset Resolution

```python
PRESETS = {
    "attention-qv": ["*.self_attn.q_proj", "*.self_attn.v_proj"],
    "attention-all": [
        "*.self_attn.q_proj", "*.self_attn.k_proj",
        "*.self_attn.v_proj", "*.self_attn.o_proj",
    ],
    "mlp": ["*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"],
    "all-linear": [
        "*.self_attn.q_proj", "*.self_attn.k_proj",
        "*.self_attn.v_proj", "*.self_attn.o_proj",
        "*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj",
    ],
}

def get_patterns(config: AdapterConfig) -> list[str]:
    if config.targets is not None:
        return config.targets
    if config.preset is not None:
        if config.preset not in PRESETS:
            raise ValueError(f"Unknown preset '{config.preset}'. Available: {list(PRESETS.keys())}")
        return PRESETS[config.preset]
    raise ValueError("No adapter targets specified.")
```

### LoRA Application

After resolving targets:

```python
from mlx.utils import tree_unflatten

def apply_lora(model, targets: list[tuple[str, nn.Module]], config: AdapterConfig):
    lora_layers = []
    for name, module in targets:
        if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            lora_module = LoRALinear.from_base(module, r=config.rank, scale=config.scale, dropout=config.dropout)
        elif isinstance(module, (nn.Embedding, nn.QuantizedEmbedding)):
            lora_module = LoRAEmbedding.from_base(module, r=config.rank, scale=config.scale, dropout=config.dropout)
        else:
            raise ValueError(f"Cannot apply LoRA to {type(module).__name__} at '{name}'.")
        lora_layers.append((name, lora_module))

    model.update_modules(tree_unflatten(lora_layers))
    return model
```

Log the number of parameters converted and the module paths.

---

## 9. Trainer Implementation Rules

### Trainer Class

```python
class Trainer:
    def __init__(self, model, config: TrainingConfig, train_dataset, val_dataset, callbacks=None):
        self.model = model
        self.config = config
        self.callbacks = CallbackList(callbacks or [])
        self.optimizer = build_optimizer(config.training, model)
        self.scheduler = build_scheduler(config.training)
        self.checkpoint_manager = CheckpointManager(config)
        self.state = TrainState(
            step=0, epoch=0, trained_tokens=0,
            best_val_loss=float("inf"),
            rng_seed=config.training.seed,
        )

    def fit(self) -> TrainState:
        ...
```

### Loss Function

```python
def loss_fn(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = (steps >= lengths[:, 0:1]) & (steps < lengths[:, 1:2])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.sum() / ntoks, ntoks
```

Note the strict less-than (`<`) for the upper bound on the loss mask. See V0_DESIGN_FREEZE.md §2.2 for the batch contract.

### Compiled Step Function

```python
# compile_state is the live computation state tracked by mx.compile.
# It is NOT the same as self.state (TrainState) or checkpoint state.
# See V0_DESIGN_FREEZE.md §5 for the distinction.
compile_state = [model.state, optimizer.state, mx.random.state]

if not config.runtime.eager:
    @partial(mx.compile, inputs=compile_state, outputs=compile_state)
    def step(batch, lengths, prev_grad, do_update):
        (loss, ntoks), grad = loss_value_and_grad(model, batch, lengths)
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
else:
    def step(batch, lengths, prev_grad, do_update):
        # identical logic, no @mx.compile
        ...
```

### Training Loop

```python
def fit(self):
    if mx.metal.is_available():
        mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

    mx.random.seed(self.config.training.seed)
    self.callbacks.on_train_begin(self.state)

    grad_accum = None
    losses = 0.0
    n_tokens = 0
    steps_since_report = 0
    report_start_time = time.perf_counter()

    for it, (batch, lengths) in zip(
        range(1, self.config.training.num_iters + 1),
        iterate_batches(self.train_dataset, self.config),
    ):
        do_update = (it % self.config.training.grad_accumulation_steps == 0)

        # Evaluate at step 1, every steps_per_eval, and final step
        if it == 1 or it % self.config.training.steps_per_eval == 0 or it == self.config.training.num_iters:
            val_loss = self.evaluate()
            self.callbacks.on_eval_end(self.state, {"val_loss": val_loss})
            if val_loss < self.state.best_val_loss:
                self.state.best_val_loss = val_loss

        # Training step
        loss, toks, grad_accum = step(batch, lengths, grad_accum, do_update)
        mx.eval(compile_state, loss, toks, grad_accum)  # SAFE POINT

        losses += loss.item()
        n_tokens += toks.item()
        steps_since_report += 1
        self.state.step = it
        self.state.trained_tokens += toks.item()

        # Reporting
        if it % self.config.training.steps_per_report == 0 or it == self.config.training.num_iters:
            elapsed = time.perf_counter() - report_start_time
            metrics = {
                "step": it,
                "train_loss": losses / steps_since_report,
                "learning_rate": self.optimizer.learning_rate.item(),
                "tokens_per_second": n_tokens / elapsed,
                "trained_tokens": self.state.trained_tokens,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
            self.callbacks.on_step_end(self.state, metrics)
            losses = 0.0
            n_tokens = 0
            steps_since_report = 0
            report_start_time = time.perf_counter()

        # Checkpointing
        if it % self.config.training.steps_per_save == 0 or it == self.config.training.num_iters:
            self.checkpoint_manager.save(self.state, self.model, self.optimizer)
            self.callbacks.on_save(self.state, self.checkpoint_manager.last_checkpoint_dir)

        # Cooperative pause check
        if self._pause_requested.is_set():
            self.checkpoint_manager.save(self.state, self.model, self.optimizer)
            self._notify_paused()
            self._block_until_resume()

    self.callbacks.on_train_end(self.state)
    return self.state
```

### Checkpoint Save

Per V0_DESIGN_FREEZE.md §2.3, each checkpoint contains exactly three files. No `manifest.json` inside checkpoints.

```python
class CheckpointManager:
    def __init__(self, config: TrainingConfig):
        self.run_dir = Path(config.runtime.run_dir).expanduser() / run_id
        self._best_val_loss = float("inf")
        self.last_checkpoint_dir = None

    def save(self, state: TrainState, model, optimizer):
        ckpt_dir = self.run_dir / "checkpoints" / f"step-{state.step:07d}"
        tmp_dir = ckpt_dir.with_suffix(".tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Adapter weights
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(tmp_dir / "adapters.safetensors"), adapter_weights)

        # Optimizer state
        opt_state = dict(tree_flatten(optimizer.state))
        mx.save_safetensors(str(tmp_dir / "optimizer.safetensors"), opt_state)

        # Train state
        state_dict = {
            "schema_version": 1,
            "step": state.step,
            "epoch": state.epoch,
            "trained_tokens": state.trained_tokens,
            "best_val_loss": state.best_val_loss,
            "learning_rate": optimizer.learning_rate.item(),
            "rng_seed": state.rng_seed,
        }
        (tmp_dir / "state.json").write_text(json.dumps(state_dict, indent=2))

        # Atomic rename
        tmp_dir.rename(ckpt_dir)
        self.last_checkpoint_dir = ckpt_dir

        # Update best symlink
        if state.best_val_loss < self._best_val_loss:
            self._best_val_loss = state.best_val_loss
            best_link = self.run_dir / "checkpoints" / "best"
            if best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(ckpt_dir.name)

        # Retention
        self._enforce_retention()
```

### Checkpoint Load (Resume)

```python
def load(self, ckpt_dir: Path, model, optimizer) -> TrainState:
    # Validate checkpoint integrity
    required = ["adapters.safetensors", "optimizer.safetensors", "state.json"]
    for f in required:
        if not (ckpt_dir / f).exists():
            raise FileNotFoundError(f"Checkpoint missing {f} in {ckpt_dir}")

    # Load state
    state_dict = json.loads((ckpt_dir / "state.json").read_text())
    if state_dict["schema_version"] > 1:
        raise ValueError(f"Unsupported checkpoint schema version: {state_dict['schema_version']}")

    # Load adapter weights
    model.load_weights(str(ckpt_dir / "adapters.safetensors"), strict=False)

    # Load optimizer state
    opt_weights = mx.load(str(ckpt_dir / "optimizer.safetensors"))
    optimizer.state = tree_unflatten(list(opt_weights.items()))

    # Restore RNG (Tier-1: re-seed, not exact state restoration)
    mx.random.seed(state_dict["rng_seed"] + state_dict["step"])

    return TrainState(
        step=state_dict["step"],
        epoch=state_dict.get("epoch", 0),
        trained_tokens=state_dict.get("trained_tokens", 0),
        best_val_loss=state_dict.get("best_val_loss", float("inf")),
        rng_seed=state_dict["rng_seed"],
    )
```

---

## 10. Logging Rules

### JSONL Metrics File

Every run produces `logs/metrics.jsonl`. Each line is one JSON object. Two event types:

**Training metrics** (emitted every `steps_per_report` steps):

```json
{
  "event": "train",
  "step": 100,
  "train_loss": 2.345,
  "learning_rate": 1e-5,
  "tokens_per_second": 15234.5,
  "trained_tokens": 409600,
  "peak_memory_gb": 12.3,
  "timestamp": "2026-02-01T14:32:15Z"
}
```

**Validation metrics** (emitted every `steps_per_eval` steps):

```json
{
  "event": "eval",
  "step": 200,
  "val_loss": 1.987,
  "timestamp": "2026-02-01T14:35:22Z"
}
```

### Console Output

Print to stdout in human-readable format. Example:

```
Step 10/1000 | loss=2.891 | lr=1.00e-05 | tok/s=14521 | mem=11.2GB
Step 20/1000 | loss=2.754 | lr=1.00e-05 | tok/s=14893 | mem=11.2GB
Step 200/1000 | val_loss=1.987
Step 200/1000 | Saved checkpoint to ~/.lmforge/runs/.../checkpoints/step-0000200
```

### MetricsLogger Callback

```python
class MetricsLoggerCallback(Callback):
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a")

    def on_step_end(self, state, metrics):
        metrics["event"] = "train"
        metrics["timestamp"] = datetime.utcnow().isoformat() + "Z"
        self._file.write(json.dumps(metrics) + "\n")
        self._file.flush()

    def on_eval_end(self, state, metrics):
        metrics["event"] = "eval"
        metrics["step"] = state.step
        metrics["timestamp"] = datetime.utcnow().isoformat() + "Z"
        self._file.write(json.dumps(metrics) + "\n")
        self._file.flush()

    def on_train_end(self, state):
        self._file.close()
```

---

## 11. Strict Implementation Order

Engineers must implement in this order. Each step must be functional and tested before proceeding.

### Step 1: Config System

- Implement all Pydantic models in `config.py` per V0_DESIGN_FREEZE.md §2.1.
- Implement `TrainingConfig.from_yaml()`.
- Write validation tests: valid config loads, invalid config raises with clear message, `steps_per_save % grad_accumulation_steps` check works, `targets`/`preset` mutual exclusion works.

### Step 2: Data Pipeline — Prepare

- Implement format detection in `data/formats.py`.
- Implement tokenization + offset computation in `data/preprocessing.py`.
- Implement safetensors cache write/read + fingerprinting in `data/cache.py`.
- Implement `prepare()` library function.
- Implement `lmforge prepare` CLI command.
- Test: prepare a small JSONL file, verify cache contents match V0_DESIGN_FREEZE.md §2.5 shard layout, verify re-running skips tokenization.

### Step 3: Data Pipeline — Batching

- Implement sort-by-length + fixed-batch + pad-to-32 iterator in `data/batching.py`.
- Test: verify output shapes match batch contract `(B, T)` and `(B, 2)` per V0_DESIGN_FREEZE.md §2.2.

### Step 4: Model Loading

- Implement model + tokenizer loading in `models/loader.py`.
- Reuse mlx-lm's `load()` function internally or reimplement with same pattern: `_download → load_config → _get_classes → Model(args) → load_weights → eval`.
- Test: load a small model (e.g., Qwen3-0.6B), verify forward pass produces logits.

### Step 5: Adapter Targeting + LoRA

- Implement glob matching in `adapters/targeting.py`.
- Implement LoRALinear and LoRAEmbedding in `adapters/lora.py` with `from_base()` and `fuse()`.
- Implement preset resolution.
- Test: apply LoRA to a loaded model, verify only targeted modules are converted, verify forward pass still works, verify `fuse()` merges correctly.

### Step 6: Optimizer + Scheduler Factory

- Implement `build_optimizer()` and `build_scheduler()` in `trainer/optimizer.py`.
- Schedulers must be stateless functions of step number. `build_scheduler()` returns a callable or MLX schedule object that the optimizer uses.
- Test: build Adam with LR schedule, verify LR changes over steps.

### Step 7: Checkpoint Manager

- Implement atomic save/load in `trainer/checkpoint.py` per V0_DESIGN_FREEZE.md §2.3.
- Checkpoint contains exactly 3 files: `adapters.safetensors`, `optimizer.safetensors`, `state.json`.
- Implement retention policy (keep last N + best).
- Test: save a checkpoint, load it, verify all state matches.

### Step 8: Callbacks + Metrics Logger

- Implement Callback base class and CallbackList in `trainer/callbacks.py`.
- Implement MetricsLoggerCallback.
- Implement optional WandBCallback (try/except import).
- Test: run a mock training loop, verify JSONL output.

### Step 9: Trainer

- Implement `Trainer.fit()` with compiled step, callback boundaries, checkpoint saves.
- Implement `Trainer.evaluate()`.
- Implement cooperative pause.
- Test: train LoRA on a small model for 50 steps, verify loss decreases, verify checkpoint produced, verify resume continues without loss spike.

### Step 10: Run Management + Manifest

- Implement run directory creation, manifest writing, environment collection.
- Implement `train()` top-level function that orchestrates everything.
- Implement `lmforge train` CLI command.
- Test: full end-to-end run from YAML config, verify all artifacts match V0_DESIGN_FREEZE.md §2.4 layout.

### Step 11: Integration Tests

- End-to-end: `lmforge prepare` → `lmforge train` → verify checkpoint → resume → verify no loss spike.
- Config validation: verify all known-bad configs fail with clear messages.
- Adapter targeting: verify glob patterns resolve correctly for at least 2 model architectures.

---

## 12. Do Not Implement

The following are explicitly out of scope for v0. Do not write code, stubs, placeholders, abstract base classes, protocol definitions, or TODO comments for any of these:

- Inference / generation engine
- Text generation sampling (top-p, top-k, temperature)
- Serving / REST API / OpenAI compatibility
- Batch generation
- Speculative decoding
- Sequence packing
- Dynamic batching by total tokens
- Streaming datasets / HuggingFace datasets integration
- DPO, RLHF, KTO, ORPO, or any non-SFT recipe
- Full parameter fine-tuning
- DoRA adapters
- IA3, prefix tuning, or other adapter types
- Distributed / multi-device training
- Gradient checkpointing / activation recomputation
- Automatic batch size detection
- Daemon process / job queue / SQLite
- GUI backend / WebSocket / resource monitoring
- Custom Metal kernels or fused LoRA operations
- GGUF export
- Model merging (beyond LoRA fusing)
- Model conversion (HF → MLX)
- Quantization (AWQ, GPTQ, dynamic)
- Plugin / registry system for extensibility
- RecipeProtocol or any abstract recipe interface
- Multi-dataset mixing
- Evaluation harness integration (lm-eval)
- Perplexity measurement
- Upload to HuggingFace Hub

---

## 13. M7: Hugging Face Model Loading (Added 2026-02-02)

### Overview

M7 adds automatic resolution of Hugging Face model IDs to local paths. Users can specify `model.path: "Qwen/Qwen3-0.8B"` directly in configs without manual downloading.

### Resolution Layer

**Location**: `lmforge/models/resolve.py`

**Entry point**: `resolve_model(model_path, revision=None, trust_remote_code=False, token=None) → ResolvedModel`

**When it runs**: At the start of `prepare()` and `train()`, before any model loading.

### Resolution Flow

1. **Classify path** - Is it a local path or HF repo ID?
   - If local: verify existence, return as-is
   - If HF: proceed to resolution

2. **Resolve revision** (online mode only)
   - If `revision` provided: use it directly
   - If `revision=None`: call `huggingface_hub.model_info()` to get latest commit hash

3. **Download/cache**
   - Call `huggingface_hub.snapshot_download(repo_id, revision, local_files_only=offline)`
   - Uses standard HF cache: `~/.cache/huggingface/hub/`
   - Idempotent: returns cached path immediately if already downloaded

4. **Return ResolvedModel**
   - `local_path`: Absolute path to model directory
   - `resolved_revision`: Pinned commit hash (for reproducibility)
   - `resolution_metadata`: Dict for manifest

### Config Changes

**One new optional field** (backward-compatible):

```python
class ModelConfig(BaseModel):
    path: str
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None  # NEW: pin to specific HF revision
```

When `revision=None` (default): resolve to latest and record commit hash in manifest.

When `revision` is specified: use exactly that revision (for reproducibility).

### Manifest Changes

**New top-level field** `model_resolution`:

```json
{
  "model_resolution": {
    "source_id": "Qwen/Qwen3-0.8B",
    "resolved_revision": "a1b2c3d4e5f6...",
    "local_path": "/Users/user/.cache/huggingface/hub/.../snapshots/a1b2c3d",
    "is_local": false
  }
}
```

This is an **additive extension**. Existing tools that read manifests ignore unknown fields (v0 forward compatibility rule).

### Offline Mode

Respects `HF_HUB_OFFLINE=1` environment variable:

- **Online mode** (default): Calls `model_info()` to pin revision, then downloads if needed
- **Offline mode**: Skips `model_info()`, uses cached snapshot only, fails if not cached

### Cache Location

Models are cached in the **standard HuggingFace Hub cache**:

```
~/.cache/huggingface/hub/
└── models--Qwen--Qwen3-0.8B/
    ├── snapshots/
    │   └── a1b2c3d.../  # The actual model files
    └── ...
```

**No LMForge-specific model cache**. This ensures:
- No duplicate storage (models are multi-GB)
- Ecosystem compatibility (shared with mlx-lm, transformers, etc.)
- Battle-tested cache management (partial downloads, concurrent access)

### Error Handling

Clear, actionable error messages for:

- **Gated models**: Tells user to accept license and set `HF_TOKEN`
- **Repo not found**: Suggests checking model ID on huggingface.co
- **Network errors**: Suggests offline mode or local path
- **Offline cache miss**: Tells user how to download first
- **Disk space**: Reports available space and suggests changing HF_HOME

### Integration Points

Resolution is called from:

1. `lmforge.train()` - Resolves model and tokenizer before loading
2. `lmforge.prepare()` - Resolves tokenizer before tokenization

Downstream consumers (`load_model()`, `AutoTokenizer.from_pretrained()`) receive resolved local paths, not raw `model.path`.

### Contract Preservation

M7 does **not** modify any v0 frozen contracts:

- ✅ Checkpoint format unchanged (3 files)
- ✅ Batch contract unchanged
- ✅ Resume semantics unchanged
- ✅ Config schema: additive extension only (`revision` field)
- ✅ Manifest: additive field (`model_resolution`)

### What M7 Does NOT Do

- Does NOT create a LMForge-specific model cache
- Does NOT implement weight conversion (delegates to mlx_lm)
- Does NOT add a `lmforge model` CLI subcommand
- Does NOT copy model weights into run directories
- Does NOT modify checkpoint directories
- Does NOT change training loop or mx.compile behavior

### Design Document

See `M7_HF_MODEL_LOADING_DESIGN.md` for the complete design specification.

---

## 14. M8: Self-Contained Model Loading

M8 removes the `mlx-lm` dependency and makes LMForge fully self-contained for model loading.

### Why Remove mlx-lm

1. **Demo library, not production-grade**: Apple built it as examples/tutorials
2. **Unstable API**: No semver guarantees, breaks between releases
3. **Overkill**: We only need ~5% of its functionality (model loading, not generation)
4. **Dependency risk**: Adds uncertainty to LMForge's stability

### Implementation

Model loading is now self-contained in `lmforge/models/`:

```
lmforge/models/
├── resolve.py              # M7 - HF resolution (unchanged)
├── loader.py               # Self-contained loading (no mlx-lm)
├── registry.py             # Explicit model allowlist
├── _base/                  # Shared utilities
│   ├── args.py             # BaseModelArgs with from_dict()
│   ├── attention.py        # create_attention_mask(), scaled_dot_product_attention()
│   ├── rope.py             # RoPE variants (standard, Llama3, SuScaled, Yarn)
│   └── activations.py      # swiglu()
└── architectures/          # Model implementations
    ├── llama.py            # Llama, Mistral (via remap)
    └── qwen3.py            # Qwen3 family
```

### Registry Pattern

Uses an explicit allowlist instead of dynamic import:

```python
SUPPORTED_ARCHITECTURES = {
    "llama": "lmforge.models.architectures.llama",
    "qwen3": "lmforge.models.architectures.qwen3",
}

MODEL_REMAPPING = {
    "mistral": "llama",  # Mistral uses Llama architecture
}
```

### Load Flow

```python
def load_model(model_path, tokenizer_path=None, trust_remote_code=False):
    # 1. Load tokenizer via transformers
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # 2. Load config.json
    config = load_config(model_path)

    # 3. Resolve model class via registry
    Model, ModelArgs = get_model_classes(config)

    # 4. Instantiate model
    model = Model(ModelArgs.from_dict(config))

    # 5. Load weights from safetensors
    weights = load_weights(model_path)

    # 6. Sanitize and load
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))

    return model, tokenizer
```

### Supported Architectures

**Tier 1 (Day One)**:
- `llama` - Llama 2/3 family
- `mistral` - Remaps to llama
- `qwen3` - Qwen3-0.6B, 1.7B, 4B, 8B

### Error Messages

```
ValueError: Model type 'deepseek_v3' is not supported.

Supported architectures: llama, qwen3

If you need this architecture, please open an issue.
```

### What M8 Does NOT Support

- **Quantized models** (AWQ, GPTQ, bitnet) - Training uses full precision
- **KV cache** - Not needed for training
- **Generation/sampling** - Training only
- **MoE models** (DeepSeek V3, Mixtral) - Can be added if needed

### Design Document

See `M8_SELF_CONTAINED_MODEL_LOADING_DESIGN.md` for the complete design specification.
