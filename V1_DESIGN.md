# LMForge V1 Design Document

> V1 extends LMForge with inference, QLoRA, performance features, and **LMForge Studio** — a browser-based training UI for experiment tracking, model management, and interactive inference.
>
> Status: **PROPOSED**
> Date: 2026-02-06

---

## Table of Contents

- [§1. V1 Goals & Non-Goals](#1-v1-goals--non-goals)
- [§2. Core Features](#2-core-features)
  - [§2.1 Fix Resume from Checkpoint](#21-fix-resume-from-checkpoint)
  - [§2.2 Inference & Generation](#22-inference--generation)
  - [§2.3 Gemma Architecture](#23-gemma-architecture)
  - [§2.4 QLoRA (Quantized Base + LoRA)](#24-qlora-quantized-base--lora)
  - [§2.5 Sequence Packing](#25-sequence-packing)
  - [§2.6 Gradient Checkpointing](#26-gradient-checkpointing)
- [§3. LMForge Studio](#3-lmforge-studio)
  - [§3.1 Architecture](#31-architecture)
  - [§3.2 Backend API](#32-backend-api)
  - [§3.3 Frontend Design](#33-frontend-design)
  - [§3.4 Real-time Training](#34-real-time-training)
  - [§3.5 Pages & Features](#35-pages--features)
  - [§3.6 Playground (Interactive Inference)](#36-playground-interactive-inference)
- [§4. Contract Preservation](#4-contract-preservation)
- [§5. Package Layout Changes](#5-package-layout-changes)
- [§6. Implementation Plan](#6-implementation-plan)
- [§7. What V1 Does NOT Do](#7-what-v1-does-not-do)
- [§8. Design Decisions Summary](#8-design-decisions-summary)

---

## §1. V1 Goals & Non-Goals

### V1 Goals

1. **Close the training loop**: Users can train AND test fine-tuned models without switching tools.
2. **Unlock larger models**: QLoRA + gradient checkpointing reduce memory 2-4x, enabling 7B+ training on 32GB machines.
3. **Improve throughput**: Sequence packing eliminates padding waste for 2-4x speedup on short-sequence datasets.
4. **Provide visibility**: LMForge Studio gives users a visual interface to track experiments, compare runs, and interactively test models.
5. **Fix production blockers**: Resume from checkpoint works end-to-end.
6. **Expand model coverage**: Gemma architecture brings coverage to ~90% of fine-tuning workloads.

### V1 Non-Goals

| Feature | Why Not V1 |
|---------|------------|
| DPO / RLHF / KTO | Different training paradigm, large scope |
| Distributed training | Apple Silicon is single-GPU |
| DoRA / IA3 adapters | Stretch goal, not blocking |
| MoE models (Mixtral, DeepSeek V3) | High complexity, niche on Mac |
| Serving / REST API / OpenAI compat | Separate concern |
| Model conversion (HF ↔ MLX) | Delegated to external tools |
| Multi-dataset mixing | Power-user feature, v2 |
| Cloud sync / multi-user | Local-first philosophy |
| Native desktop app (Tauri/Electron) | Web app first, Tauri door open for v2 |

### V1 Guarantees

- All v0 frozen contracts (§2 of V0_DESIGN_FREEZE.md) remain unchanged
- Existing configs, checkpoints, and run directories are fully compatible
- `lmforge prepare` and `lmforge train` behavior is backward-compatible
- New features are additive extensions, not modifications

---

## §2. Core Features

### §2.1 Fix Resume from Checkpoint

**Problem**: The `--resume` flag is parsed by the CLI but never passed to `train()`. Checkpoint load code exists in `CheckpointManager.load()` but is unreachable.

**Fix**: Wire `--resume` through three layers.

#### CLI Layer (`cli/train_cmd.py`)

```python
def run_train(args):
    result = train(config=args.config, resume=args.resume)
    print(f"\nTraining complete. Final step: {result.step}")
```

#### Public API (`__init__.py`)

```python
def train(config, resume: Optional[str] = None) -> TrainState:
    """
    Args:
        config: Path to YAML config file, or TrainingConfig object.
        resume: Path to checkpoint directory to resume from.
                Example: "~/.lmforge/runs/.../checkpoints/step-0000500"
    """
    ...
    if resume:
        resume_path = Path(resume).expanduser()
        state = checkpoint_manager.load(resume_path, model, optimizer)
        # Skip to the right position in data iterator
        # (Tier-1: re-iterate from start, skip `state.step` batches)
    else:
        state = TrainState(step=0, epoch=0, ...)

    trainer = Trainer(model, config_obj, train_dataset, val_dataset,
                      callbacks=callbacks, state=state)
    return trainer.fit()
```

#### Trainer Layer (`trainer/trainer.py`)

```python
class Trainer:
    def __init__(self, model, config, train_dataset, val_dataset,
                 callbacks=None, state=None):
        self.state = state or TrainState(
            step=0, epoch=0, trained_tokens=0,
            best_val_loss=float("inf"),
            rng_seed=config.training.seed,
        )
```

#### Data Iterator Skip

On resume, the batch iterator must advance past already-trained steps:

```python
batch_iter = iterate_batches(train_dataset, config)
if resume:
    # Tier-1 resume: re-iterate and skip. Not exact (shuffle may differ)
    # but good enough for continued training.
    for _ in range(state.step):
        next(batch_iter)
```

This is Tier-1 resume per V0_DESIGN_FREEZE.md §3: we re-seed RNG and skip forward, but do not guarantee exact data ordering reproducibility.

#### Resume Validation

```python
def _validate_resume(resume_path: Path, config: TrainingConfig):
    """Validate checkpoint is compatible with current config."""
    state = json.loads((resume_path / "state.json").read_text())
    if state["schema_version"] > 1:
        raise ValueError(
            f"Checkpoint schema version {state['schema_version']} is newer than "
            f"supported version 1. Please upgrade LMForge."
        )
    if state["step"] >= config.training.num_iters:
        raise ValueError(
            f"Checkpoint is at step {state['step']} but training is configured "
            f"for {config.training.num_iters} iterations. Nothing to do."
        )
```

#### Error Messages

```
Error: Cannot resume from '~/.lmforge/runs/.../checkpoints/step-0000500'

Checkpoint is missing 'optimizer.safetensors'. Expected files:
  - adapters.safetensors
  - optimizer.safetensors
  - state.json

This may indicate a corrupted or incomplete checkpoint.
```

```
Error: Checkpoint is at step 1000 but training is configured for 1000 iterations.
Nothing to do. Increase 'num_iters' in your config to continue training.
```

---

### §2.2 Inference & Generation

**Problem**: Users train LoRA adapters but have no way to test them without switching to another tool.

**Scope**: Minimal text generation — greedy decoding and top-p/temperature sampling. No beam search, speculative decoding, or batched generation.

#### New CLI Command

```bash
# Interactive chat mode
lmforge generate --model Qwen/Qwen3-0.6B --adapter ~/.lmforge/runs/.../checkpoints/best

# Single prompt
lmforge generate --model Qwen/Qwen3-0.6B --adapter ./checkpoint \
    --prompt "What is machine learning?"

# Generation parameters
lmforge generate --model Qwen/Qwen3-0.6B --adapter ./checkpoint \
    --temperature 0.7 --top-p 0.9 --max-tokens 512
```

#### Library API

```python
from lmforge import generate

# Single generation
output = generate(
    model="Qwen/Qwen3-0.6B",
    adapter="~/.lmforge/runs/.../checkpoints/best",
    prompt="What is machine learning?",
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)
print(output.text)
print(f"Tokens generated: {output.num_tokens}")
print(f"Tokens/sec: {output.tokens_per_second}")

# Streaming generation
for token in generate(
    model="Qwen/Qwen3-0.6B",
    prompt="Explain LoRA in simple terms.",
    stream=True,
):
    print(token, end="", flush=True)
```

#### GenerationResult

```python
@dataclass
class GenerationResult:
    text: str                   # Generated text (excluding prompt)
    prompt: str                 # Input prompt
    num_tokens: int             # Number of tokens generated
    tokens_per_second: float    # Generation throughput
    finish_reason: str          # "stop" (EOS), "length" (max_tokens), "error"
```

#### GenerationConfig

```python
class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=2.0)
    seed: Optional[int] = None
```

#### Sampling Implementation

```python
# Location: lmforge/inference/sampling.py

def sample_next_token(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    """Sample next token using temperature + nucleus (top-p) sampling."""
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)

    # Apply temperature
    logits = logits / temperature

    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = mx.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
        # Remove tokens with cumulative probability above threshold
        mask = cumulative_probs - mx.softmax(sorted_logits, axis=-1) >= top_p
        sorted_logits = mx.where(mask, float("-inf"), sorted_logits)
        # Scatter back to original indices
        logits = mx.zeros_like(logits)
        logits[sorted_indices] = sorted_logits

    # Sample from distribution
    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(probs)
```

#### Generation Loop

```python
# Location: lmforge/inference/engine.py

def generate_step(model, prompt_tokens, gen_config, tokenizer):
    """Generate tokens autoregressively."""
    tokens = mx.array(prompt_tokens)[None]  # (1, T)

    # Process prompt (prefill)
    cache = create_kv_cache(model)
    logits = model(tokens, cache=cache)
    next_token = sample_next_token(logits[:, -1, :], gen_config.temperature, gen_config.top_p)

    # Autoregressive decode
    generated = []
    for _ in range(gen_config.max_tokens):
        token_id = next_token.item()
        generated.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

        next_input = next_token.reshape(1, 1)
        logits = model(next_input, cache=cache)
        next_token = sample_next_token(logits[:, -1, :], gen_config.temperature, gen_config.top_p)
        mx.eval(next_token)  # Force evaluation for streaming

        yield token_id
```

#### KV Cache

Inference requires a KV cache that the current training-only model architectures don't implement. Each architecture needs a `__call__` that optionally accepts a cache:

```python
# Added to each architecture's Attention class
class Attention(nn.Module):
    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k = self.rope(q, k, cache_offset=cache.offset if cache else 0)

        if cache is not None:
            k, v = cache.update(k, v)

        output = scaled_dot_product_attention(q, k, v, mask=mask, scale=self.scale)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))
```

```python
# Location: lmforge/inference/cache.py

class KVCache:
    """Simple KV cache for autoregressive generation."""

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0

    def update(self, keys: mx.array, values: mx.array):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset = self.keys.shape[2]
        return self.keys, self.values
```

#### Adapter Loading for Inference

```python
def load_for_inference(model_path, adapter_path=None, trust_remote_code=False):
    """Load model and optionally apply + fuse LoRA adapter for inference."""
    model, tokenizer = load_model(model_path, trust_remote_code=trust_remote_code)

    if adapter_path:
        adapter_path = Path(adapter_path).expanduser()
        # Load adapter weights
        adapter_weights = mx.load(str(adapter_path / "adapters.safetensors"))

        # Determine which modules need LoRA wrappers from weight keys
        # Keys look like: "model.layers.0.self_attn.q_proj.lora_a"
        lora_modules = _detect_lora_modules(adapter_weights)

        # Apply LoRA wrappers, load weights, then fuse
        apply_lora_from_weights(model, adapter_weights)
        fuse_lora(model)

    return model, tokenizer
```

#### Interactive Chat Mode

```python
# Location: lmforge/cli/generate_cmd.py

def run_generate_interactive(model, tokenizer, gen_config):
    """Interactive chat REPL."""
    print("LMForge Interactive Generation")
    print("Type 'quit' to exit, 'clear' to reset context.\n")

    messages = []
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() == "quit":
            break
        if user_input.strip().lower() == "clear":
            messages = []
            print("[Context cleared]\n")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        print("Assistant: ", end="", flush=True)
        generated_text = ""
        for token_id in generate_step(model, prompt_tokens, gen_config, tokenizer):
            text = tokenizer.decode([token_id])
            print(text, end="", flush=True)
            generated_text += text
        print()

        messages.append({"role": "assistant", "content": generated_text})
```

#### What Inference Does NOT Do

- No beam search (greedy + sampling is sufficient for testing fine-tunes)
- No batched generation (single prompt at a time)
- No speculative decoding
- No continuous batching
- No OpenAI-compatible API (that's a serving concern)
- No grammar-constrained generation
- No logprob output

---

### §2.3 Gemma Architecture

**Problem**: Google's Gemma 2/3 models are widely used for fine-tuning but not supported.

**Effort**: ~1 day. Gemma shares structural similarities with existing architectures but has distinctive features.

#### Architecture Differences from Llama

| Feature | Llama | Gemma |
|---------|-------|-------|
| Normalization | RMSNorm | RMSNorm + learnable offset (+1) |
| Activation | SiLU | GELU (Gemma 1) / GeGLU (Gemma 2) |
| Embeddings | Separate lm_head | Tied (embed + lm_head share weights) |
| Head dim | hidden_size / n_heads | Explicit `head_dim` in config |
| Attention | Standard | Soft-capping (Gemma 2) + sliding window |
| QK norm | No | Optional (Gemma 2) |
| Post-norm | No | Gemma 2 adds post-attention and post-FFN RMSNorm |

#### Implementation

```python
# Location: lmforge/models/architectures/gemma.py

class GemmaModelArgs(BaseModelArgs):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    # Gemma 2 features
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    sliding_window: Optional[int] = None
    query_pre_attn_scalar: Optional[float] = None

    @classmethod
    def from_dict(cls, config: dict) -> "GemmaModelArgs":
        # Map HF config keys to our fields
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})
```

```python
class GemmaAttention(nn.Module):
    def __call__(self, x, mask=None, cache=None):
        ...
        # Gemma 2: Soft attention capping
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = mx.tanh(scores)
            scores = scores * self.attn_logit_softcapping
        ...
```

```python
class GemmaRMSNorm(nn.Module):
    """Gemma RMSNorm with +1 offset (weight acts as offset from 1.0)."""
    def __call__(self, x):
        norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * norm * (1 + self.weight)  # Note: (1 + weight), not just weight
```

#### Registry Update

```python
SUPPORTED_ARCHITECTURES = {
    "llama": "lmforge.models.architectures.llama",
    "phi3": "lmforge.models.architectures.phi3",
    "qwen3": "lmforge.models.architectures.qwen3",
    "gemma": "lmforge.models.architectures.gemma",   # NEW
    "gemma2": "lmforge.models.architectures.gemma",  # NEW (same module)
}
```

#### Models Covered

| Model | Params | HF ID |
|-------|--------|-------|
| Gemma 2B | 2.5B | google/gemma-2b |
| Gemma 7B | 8.5B | google/gemma-7b |
| Gemma 2 2B | 2.6B | google/gemma-2-2b |
| Gemma 2 9B | 9.2B | google/gemma-2-9b |
| Gemma 2 27B | 27.2B | google/gemma-2-27b |
| Gemma 3 1B | 1.0B | google/gemma-3-1b-pt |
| Gemma 3 4B | 4.3B | google/gemma-3-4b-pt |
| Gemma 3 12B | 12.2B | google/gemma-3-12b-pt |
| Gemma 3 27B | 27.4B | google/gemma-3-27b-pt |

---

### §2.4 QLoRA (Quantized Base + LoRA)

**Problem**: Full-precision LoRA on a 7B model requires ~28GB RAM. On a 32GB Mac, this leaves almost no headroom. QLoRA quantizes the frozen base model to 4-bit, reducing memory by ~60-70%.

**Key Insight**: Only the **frozen base weights** are quantized. LoRA adapter weights (A, B matrices) remain in full precision (float16/float32). Gradients flow through the dequantized base weights.

#### Memory Impact

| Model | Full Precision | QLoRA (4-bit) | Savings |
|-------|---------------|---------------|---------|
| Qwen3-0.6B | ~2.4 GB | ~0.8 GB | 67% |
| Llama-3.2-3B | ~12 GB | ~4 GB | 67% |
| Llama-3.1-8B | ~32 GB | ~10 GB | 69% |
| Gemma-2-27B | ~108 GB | ~34 GB | 69% |

#### Config Extension

```python
class QuantizationConfig(BaseModel):
    """Quantization settings for QLoRA training."""
    model_config = ConfigDict(extra="forbid")

    bits: int = Field(default=4, description="Quantization bits (4 or 8)")
    group_size: int = Field(default=64, description="Quantization group size")

    @field_validator("bits")
    @classmethod
    def validate_bits(cls, v):
        if v not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {v}")
        return v

    @field_validator("group_size")
    @classmethod
    def validate_group_size(cls, v):
        if v not in (32, 64, 128):
            raise ValueError(f"group_size must be 32, 64, or 128, got {v}")
        return v
```

```python
class ModelConfig(BaseModel):
    path: str
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    quantization: Optional[QuantizationConfig] = None  # NEW
```

#### Example Config

```yaml
model:
  path: meta-llama/Llama-3.1-8B-Instruct
  quantization:
    bits: 4
    group_size: 64

adapter:
  preset: attention-qv
  rank: 16
  scale: 32.0
```

#### Quantization Implementation

```python
# Location: lmforge/models/quantize.py

def quantize_model(model: nn.Module, config: QuantizationConfig) -> nn.Module:
    """Quantize all Linear layers in the model to reduce memory.

    Only freezes and quantizes the base model weights.
    LoRA adapters (applied after quantization) remain in full precision.
    """
    nn.quantize(
        model,
        bits=config.bits,
        group_size=config.group_size,
        # class_predicate filters which modules to quantize.
        # Default: all nn.Linear. Excludes lm_head if tied to embeddings.
    )
    return model
```

#### Integration with LoRA

The order matters:

```python
# In train():
model, tokenizer = load_model(model_path)

# Step 1: Quantize base model (reduces memory)
if config.model.quantization:
    quantize_model(model, config.model.quantization)

# Step 2: Apply LoRA on top of quantized model
# LoRALinear.from_base() already handles QuantizedLinear
targets = resolve_targets(model, patterns, config.adapter.num_layers)
apply_lora(model, targets, config.adapter)

# LoRA's from_base() already works with QuantizedLinear:
# if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
#     lora_module = LoRALinear.from_base(module, ...)
```

#### What Quantization Does NOT Do

- Does NOT support pre-quantized model weights (GPTQ, AWQ formats)
- Does NOT quantize LoRA adapter weights (those stay in full precision)
- Does NOT quantize during inference (models are already small enough)
- Does NOT support mixed-precision quantization (uniform bits across all layers)
- Does NOT change the checkpoint format (adapters are still float16/32)

---

### §2.5 Sequence Packing

**Problem**: Short sequences (e.g., Alpaca average ~200 tokens) padded to max length waste 80%+ of compute. A batch of 4 samples at 200 tokens padded to 2048 means 7,392 padding tokens doing nothing.

**Solution**: Pack multiple sequences into a single row, separated by EOS tokens, with an attention mask that prevents cross-sequence attention.

#### Impact

| Dataset | Avg Length | Max Length | Padding Waste (No Packing) | Speedup (With Packing) |
|---------|-----------|-----------|---------------------------|----------------------|
| Alpaca | ~200 | 2048 | ~90% | ~3-5x |
| ShareGPT | ~500 | 2048 | ~75% | ~2-3x |
| Long-form | ~1500 | 2048 | ~25% | ~1.2x |

#### Config Extension

```python
class DataConfig(BaseModel):
    # ... existing fields ...
    packing: bool = False  # NEW: Enable sequence packing
```

```yaml
data:
  train: ./train.jsonl
  packing: true
  max_seq_length: 2048
```

#### Packing Algorithm

```python
# Location: lmforge/data/packing.py

def pack_sequences(
    samples: list[dict],
    max_seq_length: int,
    pad_multiple: int = 32,
) -> list[PackedBatch]:
    """Pack multiple samples into fixed-length rows using first-fit-decreasing.

    Each packed row contains:
      - tokens: concatenated token sequences (with EOS between them)
      - segment_ids: integer ID per token indicating which sample it belongs to
      - offsets: list of (prompt_offset, total_length) per sample in the row
    """
    # Sort by length descending (greedy bin packing)
    indexed = sorted(enumerate(samples), key=lambda x: len(x[1]["tokens"]), reverse=True)

    bins = []  # Each bin is a list of (sample_idx, tokens, offset, length)
    bin_lengths = []

    for orig_idx, sample in indexed:
        tokens = sample["tokens"]
        offset = sample["offset"]
        length = len(tokens)

        # Find first bin with room
        placed = False
        for i, bin_len in enumerate(bin_lengths):
            if bin_len + length <= max_seq_length:
                bins[i].append((orig_idx, tokens, offset, length))
                bin_lengths[i] += length
                placed = True
                break

        if not placed:
            bins.append([(orig_idx, tokens, offset, length)])
            bin_lengths.append(length)

    # Convert bins to packed batches
    packed = []
    for bin_samples in bins:
        packed_tokens = []
        segment_ids = []
        offsets = []

        running_offset = 0
        for seg_id, (_, tokens, offset, length) in enumerate(bin_samples):
            packed_tokens.extend(tokens)
            segment_ids.extend([seg_id] * length)
            offsets.append((running_offset + offset, running_offset + length))
            running_offset += length

        # Pad to multiple
        pad_len = _round_up(len(packed_tokens), pad_multiple) - len(packed_tokens)
        packed_tokens.extend([0] * pad_len)
        segment_ids.extend([-1] * pad_len)  # -1 = padding

        packed.append(PackedBatch(
            tokens=packed_tokens,
            segment_ids=segment_ids,
            sample_offsets=offsets,
        ))

    return packed
```

#### Loss Function Update

With packing, the loss mask must also prevent loss computation across sequence boundaries:

```python
def loss_fn_packed(model, batch, segment_ids, offsets):
    """Loss function for packed sequences.

    batch:       (B, T) token IDs
    segment_ids: (B, T) segment membership (-1 = padding)
    offsets:     (B, max_segments, 2) per-segment (prompt_end, total_end)
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    seg_in = segment_ids[:, :-1]
    seg_out = segment_ids[:, 1:]

    logits = model(inputs, segment_ids=seg_in)  # Attention uses segment_ids for masking

    # Mask: only compute loss where input/output are same segment AND past prompt
    same_segment = (seg_in == seg_out) & (seg_out >= 0)
    past_prompt = _compute_prompt_mask(seg_out, offsets)
    mask = same_segment & past_prompt

    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.sum() / ntoks, ntoks
```

#### Attention Masking for Packing

Packed sequences require a block-diagonal attention mask to prevent cross-sequence attention:

```python
def create_packed_attention_mask(segment_ids: mx.array) -> mx.array:
    """Create attention mask where tokens can only attend within their segment.

    Args:
        segment_ids: (B, T) with segment indices, -1 for padding

    Returns:
        mask: (B, 1, T, T) boolean mask, True = allowed attention
    """
    # Same segment mask: seg[i] == seg[j] and seg[i] >= 0
    seg = segment_ids[:, None, :]       # (B, 1, T)
    seg_t = segment_ids[:, :, None]     # (B, T, 1)
    same_seg = (seg == seg_t) & (seg >= 0)

    # Combine with causal mask
    T = segment_ids.shape[1]
    causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    return (same_seg[:, None, :, :] & causal[None, None, :, :])
```

#### Fallback

When `packing: false` (default), the existing batching logic is unchanged. Packing is opt-in.

---

### §2.6 Gradient Checkpointing

**Problem**: During backpropagation, all intermediate activations are stored in memory. For large models, this dominates memory usage. Gradient checkpointing recomputes activations during the backward pass instead of storing them.

**Tradeoff**: ~30% more compute for ~40-60% less activation memory.

#### Config Extension

```python
class TrainingParams(BaseModel):
    # ... existing fields ...
    gradient_checkpointing: bool = False  # NEW
```

```yaml
training:
  gradient_checkpointing: true
  batch_size: 4
```

#### Implementation

MLX provides `mx.checkpoint()` for this pattern:

```python
# Location: lmforge/trainer/trainer.py (during model setup)

def enable_gradient_checkpointing(model: nn.Module):
    """Wrap each transformer layer's __call__ with mx.checkpoint().

    This recomputes activations during backward pass instead of storing them,
    reducing memory at the cost of ~30% more compute.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            layer._original_call = layer.__call__
            layer.__call__ = mx.checkpoint(layer.__call__)
```

#### Integration

```python
# In train():
if config.training.gradient_checkpointing:
    enable_gradient_checkpointing(model)
    # Log the memory savings
    print(f"Gradient checkpointing enabled for {num_layers} transformer layers")
```

#### When to Use

| Scenario | Recommendation |
|----------|---------------|
| QLoRA + small model (≤3B) | Not needed |
| QLoRA + medium model (7-8B) | Recommended |
| Full precision + any model | Recommended |
| Short sequences (≤512) | Marginal benefit |
| Long sequences (≥2048) | Strong benefit |

---

## §3. LMForge Studio

### §3.1 Architecture

LMForge Studio is a browser-based UI started by `lmforge studio`. It is a **read-write window into the existing `~/.lmforge/` directory structure** — not a separate system.

#### Design Principles

1. **No database**: The filesystem IS the database. Runs, checkpoints, metrics, and caches are already structured files.
2. **No daemon**: Studio starts on demand and stops on Ctrl+C. No background process.
3. **Library-first**: Studio backend calls the same `lmforge.train()`, `lmforge.prepare()`, and `lmforge.generate()` functions as the CLI.
4. **Additive**: Studio is an optional dependency. Core LMForge works without it.
5. **Tauri-ready**: Frontend is a standalone SPA that talks to the backend over HTTP/WebSocket. Can be wrapped in Tauri later with zero frontend changes.

#### System Diagram

```
┌──────────────────────────────────────────────────────────┐
│                     Browser (React SPA)                   │
│  ┌──────────┬──────────┬──────────┬──────────┬─────────┐ │
│  │Dashboard │Experiments│ Models  │ Datasets │Playground│ │
│  └──────────┴──────────┴──────────┴──────────┴─────────┘ │
│          │ HTTP/REST              │ WebSocket              │
└──────────┼────────────────────────┼────────────────────────┘
           │                        │
┌──────────┴────────────────────────┴────────────────────────┐
│              FastAPI Server (lmforge studio)                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  REST API        WebSocket Hub       Static Files    │  │
│  │  /api/runs/*     /ws/training        /static/*       │  │
│  │  /api/models/*   /ws/inference                       │  │
│  │  /api/datasets/*                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│  ┌──────────────────────┴──────────────────────────────┐   │
│  │              Service Layer                           │   │
│  │  RunService  ModelService  DatasetService  Training  │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                          │                                  │
│           Calls lmforge library functions directly          │
│           (prepare, train, generate, load_model, etc.)      │
└─────────────────────────┬──────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │   ~/.lmforge/          │
              │   ├── runs/            │  ← Experiments
              │   ├── cache/           │  ← Datasets
              │   └── studio.yaml      │  ← Studio settings
              │                        │
              │   ~/.cache/huggingface/ │  ← Models (HF cache)
              └────────────────────────┘
```

#### Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Backend framework | **FastAPI** | Async, WebSocket support, Pydantic integration (already a dependency) |
| Frontend framework | **Vite + React 19 + TypeScript** | Fast builds, modern, largest ecosystem |
| UI components | **Tailwind CSS + shadcn/ui** | Clean design system, Apple-like aesthetic |
| Charts | **Recharts** | Best React charting library, good for loss curves |
| Real-time | **WebSocket** (native) | Live training metrics, streaming inference |
| State management | **React Query (TanStack Query)** | Server state caching, auto-refresh, polling |
| Routing | **React Router v7** | Standard SPA routing |
| Build | **Vite** | Fast dev server, optimized production builds |
| Icons | **Lucide React** | Clean, consistent icon set |

#### Optional Dependency

Studio adds these to `pyproject.toml` as an optional dependency group:

```toml
[project.optional-dependencies]
studio = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "websockets>=12.0",
]
```

Install: `pip install lmforge[studio]`

The frontend is pre-built and shipped as static files inside the Python package. No Node.js required at runtime.

---

### §3.2 Backend API

All API routes are prefixed with `/api/v1/`. The API is internal (localhost only) and not versioned for external consumers.

#### Experiments API

```
GET    /api/v1/runs                    List all runs (sorted by date, newest first)
GET    /api/v1/runs/:run_id            Get run details (config, manifest, state)
GET    /api/v1/runs/:run_id/metrics    Get metrics (parsed from metrics.jsonl)
GET    /api/v1/runs/:run_id/config     Get training config
GET    /api/v1/runs/:run_id/checkpoints List checkpoints for a run
DELETE /api/v1/runs/:run_id            Delete a run and all artifacts
```

**Run list response:**

```json
{
  "runs": [
    {
      "run_id": "20260205-143215-sft-qwen3-a1b2",
      "status": "completed",
      "model": "Qwen/Qwen3-0.6B",
      "created_at": "2026-02-05T14:32:15Z",
      "num_iters": 1000,
      "current_step": 1000,
      "best_val_loss": 1.234,
      "train_loss": 1.456,
      "tokens_per_second": 1180.5,
      "peak_memory_gb": 18.2,
      "adapter_preset": "attention-qv",
      "duration_seconds": 3600
    }
  ]
}
```

**Metrics response:**

```json
{
  "train": [
    {"step": 10, "train_loss": 2.891, "learning_rate": 1e-5, "tokens_per_second": 14521},
    {"step": 20, "train_loss": 2.754, "learning_rate": 1e-5, "tokens_per_second": 14893}
  ],
  "eval": [
    {"step": 100, "val_loss": 1.987},
    {"step": 200, "val_loss": 1.654}
  ]
}
```

#### Run Service Implementation

```python
# Location: lmforge/studio/services/run_service.py

class RunService:
    """Discovers and reads runs from ~/.lmforge/runs/."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def list_runs(self) -> list[RunSummary]:
        runs = []
        for run_path in sorted(self.run_dir.iterdir(), reverse=True):
            if not run_path.is_dir():
                continue
            summary = self._read_run_summary(run_path)
            if summary:
                runs.append(summary)
        return runs

    def _read_run_summary(self, run_path: Path) -> Optional[RunSummary]:
        config_path = run_path / "config.yaml"
        manifest_path = run_path / "manifest.json"
        metrics_path = run_path / "logs" / "metrics.jsonl"

        if not config_path.exists():
            return None

        config = yaml.safe_load(config_path.read_text())
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

        # Parse last metrics line for current state
        last_train = self._last_metric(metrics_path, "train")
        last_eval = self._last_metric(metrics_path, "eval")

        # Determine status
        status = self._infer_status(run_path, config, last_train)

        return RunSummary(
            run_id=run_path.name,
            status=status,
            model=config.get("model", {}).get("path", "unknown"),
            created_at=manifest.get("created_at", ""),
            num_iters=config.get("training", {}).get("num_iters", 0),
            current_step=last_train.get("step", 0) if last_train else 0,
            best_val_loss=last_eval.get("val_loss") if last_eval else None,
            train_loss=last_train.get("train_loss") if last_train else None,
            tokens_per_second=last_train.get("tokens_per_second") if last_train else None,
            peak_memory_gb=last_train.get("peak_memory_gb") if last_train else None,
        )

    def _infer_status(self, run_path, config, last_train) -> str:
        """Infer run status from filesystem state."""
        if last_train and last_train.get("step", 0) >= config.get("training", {}).get("num_iters", 0):
            return "completed"
        # Check if metrics file was recently modified (within 60s = likely still running)
        metrics_path = run_path / "logs" / "metrics.jsonl"
        if metrics_path.exists():
            mtime = metrics_path.stat().st_mtime
            if time.time() - mtime < 60:
                return "running"
        if last_train:
            return "stopped"
        return "pending"
```

#### Models API

```
GET    /api/v1/models                  List all downloaded models
GET    /api/v1/models/:model_id        Get model details (architecture, params, size)
POST   /api/v1/models/download         Download a model from HuggingFace
GET    /api/v1/models/supported        List supported architectures
```

**Model list response:**

```json
{
  "models": [
    {
      "model_id": "Qwen/Qwen3-0.6B",
      "local_path": "/Users/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/abc123/",
      "architecture": "qwen3",
      "supported": true,
      "size_gb": 1.2,
      "num_parameters": "596M",
      "runs": ["20260205-143215-sft-qwen3-a1b2", "20260204-091022-sft-qwen3-c3d4"]
    }
  ]
}
```

#### Model Service Implementation

```python
# Location: lmforge/studio/services/model_service.py

class ModelService:
    """Discovers models from HuggingFace cache and links to runs."""

    def __init__(self, hf_cache_dir: Path, run_service: RunService):
        self.hf_cache_dir = hf_cache_dir
        self.run_service = run_service

    def list_models(self) -> list[ModelSummary]:
        models = []
        models_dir = self.hf_cache_dir / "hub"
        if not models_dir.exists():
            return []

        for model_dir in models_dir.iterdir():
            if not model_dir.name.startswith("models--"):
                continue

            model_id = model_dir.name.replace("models--", "").replace("--", "/")
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists():
                continue

            # Get latest snapshot
            snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if not snapshots:
                continue

            local_path = snapshots[0]
            config = self._read_config(local_path)
            architecture = config.get("model_type", "unknown") if config else "unknown"
            supported = is_supported(architecture)

            models.append(ModelSummary(
                model_id=model_id,
                local_path=str(local_path),
                architecture=architecture,
                supported=supported,
                size_gb=self._dir_size_gb(local_path),
                num_parameters=config.get("num_parameters", "unknown") if config else "unknown",
                runs=self._find_runs_using_model(model_id),
            ))

        return models
```

#### Datasets API

```
GET    /api/v1/datasets                List all prepared datasets
GET    /api/v1/datasets/:fingerprint   Get dataset details (meta.json contents)
GET    /api/v1/datasets/:fingerprint/preview  Preview first N samples
POST   /api/v1/datasets/prepare        Prepare a dataset (triggers lmforge prepare)
DELETE /api/v1/datasets/:fingerprint   Delete a cached dataset
```

#### Training API

```
POST   /api/v1/training/start          Start a new training run
POST   /api/v1/training/:run_id/stop   Request cooperative stop
POST   /api/v1/training/:run_id/resume Resume from checkpoint
GET    /api/v1/training/active          List currently active training runs
```

**Start training request:**

```json
{
  "config": {
    "model": {"path": "Qwen/Qwen3-0.6B"},
    "adapter": {"preset": "attention-qv", "rank": 16},
    "data": {"train": "./train.jsonl"},
    "training": {"num_iters": 1000, "batch_size": 2}
  }
}
```

**Start training response:**

```json
{
  "run_id": "20260206-103045-sft-qwen3-e5f6",
  "status": "started",
  "ws_url": "/ws/training/20260206-103045-sft-qwen3-e5f6"
}
```

#### Inference API

```
POST   /api/v1/inference/generate      Generate text (non-streaming)
WS     /ws/inference                   Streaming generation via WebSocket
POST   /api/v1/inference/load          Load a model + adapter into memory
POST   /api/v1/inference/unload        Unload model from memory
GET    /api/v1/inference/status         Current loaded model info
```

**Generate request:**

```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "adapter": "~/.lmforge/runs/.../checkpoints/best"
}
```

---

### §3.3 Frontend Design

#### Design Language

LMForge Studio follows a **clean, information-dense** design language inspired by Linear, Vercel, and Apple developer tools:

- **Color scheme**: Dark mode default, with light mode option
- **Typography**: System font stack (SF Pro on macOS), monospace for metrics/code
- **Spacing**: Generous whitespace, clear visual hierarchy
- **Interactions**: Minimal animations, instant feedback, no loading spinners >200ms

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│  ┌─────┐  LMForge Studio                    [Dark/Light]  │
│  │Logo │                                                    │
│  └─────┘                                                    │
├──────────┬─────────────────────────────────────────────────┤
│          │                                                  │
│ Dashboard│            Main Content Area                     │
│          │                                                  │
│ Exper-   │   ┌─────────────────────────────────────────┐   │
│ iments   │   │                                         │   │
│          │   │                                         │   │
│ Models   │   │     Page-specific content               │   │
│          │   │                                         │   │
│ Datasets │   │                                         │   │
│          │   │                                         │   │
│ Play-    │   │                                         │   │
│ ground   │   │                                         │   │
│          │   └─────────────────────────────────────────┘   │
│          │                                                  │
│ Settings │                                                  │
│          │                                                  │
└──────────┴─────────────────────────────────────────────────┘
```

- **Sidebar**: Fixed 240px, collapsible to icons on narrow screens
- **Main area**: Fluid width, max-width 1400px, centered
- **No header bar** beyond the logo row — maximize vertical content space

---

### §3.4 Real-time Training

#### WebSocket Protocol

```
WS /ws/training/:run_id
```

**Server → Client messages:**

```json
{"type": "metric", "data": {"event": "train", "step": 100, "train_loss": 2.345, ...}}
{"type": "metric", "data": {"event": "eval", "step": 200, "val_loss": 1.987}}
{"type": "checkpoint", "data": {"step": 200, "path": "checkpoints/step-0000200"}}
{"type": "status", "data": {"status": "running", "step": 150}}
{"type": "status", "data": {"status": "completed", "final_step": 1000}}
{"type": "error", "data": {"message": "CUDA out of memory", "step": 342}}
```

**Client → Server messages:**

```json
{"type": "stop"}
{"type": "pause"}
{"type": "resume"}
```

#### Two Monitoring Modes

**Mode 1: Passive (CLI-started runs)**

Studio watches `metrics.jsonl` via periodic polling (1s interval):

```python
# Location: lmforge/studio/services/metrics_watcher.py

class MetricsWatcher:
    """Watches metrics.jsonl for new lines and broadcasts via WebSocket."""

    def __init__(self, metrics_path: Path, ws_manager: WebSocketManager):
        self.metrics_path = metrics_path
        self.ws_manager = ws_manager
        self._last_position = 0

    async def poll(self):
        """Called every 1 second."""
        if not self.metrics_path.exists():
            return

        with open(self.metrics_path) as f:
            f.seek(self._last_position)
            new_lines = f.readlines()
            self._last_position = f.tell()

        for line in new_lines:
            metric = json.loads(line.strip())
            await self.ws_manager.broadcast(
                {"type": "metric", "data": metric}
            )
```

**Mode 2: Active (Studio-started runs)**

Studio starts training as a subprocess and captures output:

```python
# Location: lmforge/studio/services/training_service.py

class TrainingService:
    """Manages training runs started from Studio."""

    def __init__(self, run_dir: Path):
        self.active_runs: dict[str, TrainingProcess] = {}

    async def start_training(self, config: dict) -> str:
        """Start a training run in a subprocess."""
        # Write config to temp file
        config_path = self._write_temp_config(config)

        # Start training in subprocess
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "lmforge.cli.main", "train",
            "--config", str(config_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        run_id = self._extract_run_id(config)
        self.active_runs[run_id] = TrainingProcess(
            process=process,
            config=config,
            started_at=datetime.utcnow(),
        )

        return run_id

    async def stop_training(self, run_id: str):
        """Send SIGINT for cooperative shutdown."""
        if run_id in self.active_runs:
            self.active_runs[run_id].process.send_signal(signal.SIGINT)
```

---

### §3.5 Pages & Features

#### Page 1: Dashboard

**Purpose**: At-a-glance overview of recent activity.

**Content**:
- Active training runs (if any) with live progress bar
- Last 5 completed runs (mini-table)
- Quick stats: total runs, total training hours, models downloaded
- Quick actions: "New Training Run", "Open Playground"

#### Page 2: Experiments

**Purpose**: Browse, compare, and manage training runs.

**Views**:

**List view** (default):
```
┌──────────────────────────────────────────────────────────┐
│ Experiments                          [+ New Run] [Filter]│
├──────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────┐   │
│ │ ● run-20260205-143215    Qwen3-0.6B    Completed   │   │
│ │   attention-qv r=16  │  Loss: 1.23  │  1180 tok/s  │   │
│ │   1000/1000 steps     │  2h 14m      │  18.2 GB     │   │
│ └────────────────────────────────────────────────────┘   │
│ ┌────────────────────────────────────────────────────┐   │
│ │ ◉ run-20260206-091022    Llama-3.2-3B  Running     │   │
│ │   all-linear r=8     │  Loss: 2.45  │  892 tok/s   │   │
│ │   ████████░░  342/1000│  0h 48m      │  24.1 GB     │   │
│ └────────────────────────────────────────────────────┘   │
```

**Run detail view** (click into a run):
```
┌──────────────────────────────────────────────────────────┐
│ ← Back    run-20260205-143215         [Resume] [Delete]  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─ Loss Curve ───────────────────────────────────────┐  │
│  │          ╲                                         │  │
│  │           ╲___                                     │  │
│  │               ╲____                                │  │
│  │  ── Train          ╲_______╲___________            │  │
│  │  -- Val                                            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─ Metrics ──────────┐  ┌─ Config ──────────────────┐  │
│  │ Final Loss:  1.234  │  │ Model: Qwen/Qwen3-0.6B   │  │
│  │ Best Val:    1.198  │  │ Adapter: attention-qv     │  │
│  │ Throughput:  1180/s │  │ Rank: 16, Scale: 32.0     │  │
│  │ Total Time:  2h 14m │  │ LR: 1e-5 (cosine decay)  │  │
│  │ Peak Mem:    18.2GB │  │ Batch: 2 × 8 accum       │  │
│  │ Tokens:      4.3M   │  │ Steps: 1000               │  │
│  └─────────────────────┘  └───────────────────────────┘  │
│                                                          │
│  ┌─ Checkpoints ──────────────────────────────────────┐  │
│  │  ★ step-0000800 (best)  │ val_loss: 1.198          │  │
│  │    step-0000900          │ val_loss: 1.212          │  │
│  │    step-0001000 (final)  │ val_loss: 1.234          │  │
│  │                     [Load in Playground] [Export]   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─ Environment ──────────────────────────────────────┐  │
│  │  Chip: Apple M4 Pro  │  Memory: 48 GB              │  │
│  │  MLX: 0.18.1         │  LMForge: 1.0.0             │  │
│  └────────────────────────────────────────────────────┘  │
```

**Compare view** (select 2-3 runs):
- Overlaid loss curves on same chart
- Side-by-side config diff (highlighted differences)
- Metrics comparison table

#### Page 3: Models

**Purpose**: Browse downloaded models, download new ones.

```
┌──────────────────────────────────────────────────────────┐
│ Models                                       [Download]  │
├──────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────┐   │
│ │  Qwen/Qwen3-0.6B                    qwen3  ✓      │   │
│ │  596M params  │  1.2 GB  │  3 runs                 │   │
│ │                          [Open in Playground]       │   │
│ └────────────────────────────────────────────────────┘   │
│ ┌────────────────────────────────────────────────────┐   │
│ │  meta-llama/Llama-3.2-3B-Instruct    llama  ✓     │   │
│ │  3.2B params  │  6.4 GB  │  1 run                  │   │
│ └────────────────────────────────────────────────────┘   │
│ ┌────────────────────────────────────────────────────┐   │
│ │  deepseek-ai/DeepSeek-V3             unknown ✗     │   │
│ │  685B params  │  234 GB  │  Not supported           │   │
│ └────────────────────────────────────────────────────┘   │
```

**Download modal**:
- Text input for HuggingFace model ID
- Shows model card info (fetched from HF API)
- Architecture compatibility check before downloading
- Download progress bar

#### Page 4: Datasets

**Purpose**: Browse prepared datasets and their statistics.

```
┌──────────────────────────────────────────────────────────┐
│ Datasets                                    [Prepare]    │
├──────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────┐   │
│ │  alpaca_train.jsonl                   chat format  │   │
│ │  49,002 samples  │  4.3M tokens  │  2 shards      │   │
│ │  Lengths: min=12  avg=87  max=1024                 │   │
│ │  ┌─ Token Length Distribution ──────────────────┐  │   │
│ │  │  ████████████████                            │  │   │
│ │  │  ████████████                                │  │   │
│ │  │  ████████                                    │  │   │
│ │  │  ████                                        │  │   │
│ │  │  ██                                          │  │   │
│ │  └──────────────────────────────────────────────┘  │   │
│ └────────────────────────────────────────────────────┘   │
```

**Dataset detail view**:
- Full statistics from `meta.json`
- Token length histogram
- Sample preview (first 20 samples rendered as chat bubbles)
- Linked training runs that used this dataset

#### Page 5: Settings

- Default run directory path
- Default HF cache path
- Studio port configuration
- Theme (dark/light)
- Saved as `~/.lmforge/studio.yaml`

---

### §3.6 Playground (Interactive Inference)

**Purpose**: Test fine-tuned models interactively. This is the payoff — users see the result of their training.

**Depends on**: §2.2 (Inference & Generation)

```
┌──────────────────────────────────────────────────────────┐
│ Playground                                               │
├─────────────────────────────┬────────────────────────────┤
│                             │  Model: [Qwen/Qwen3-0.6B▾]│
│   Chat Area                 │  Adapter: [best (run-7) ▾] │
│                             │                            │
│  ┌────────────────────────┐ │  Temperature: [0.7    ]    │
│  │ You: What is LoRA?     │ │  Top-p:       [0.9    ]    │
│  └────────────────────────┘ │  Max tokens:  [512    ]    │
│  ┌────────────────────────┐ │                            │
│  │ Assistant: LoRA stands │ │  ── Throughput ──          │
│  │ for Low-Rank Adaptation│ │  42.3 tok/s                │
│  │ ...                    │ │  128 tokens generated      │
│  └────────────────────────┘ │                            │
│                             │  [Clear] [Export Chat]     │
│  ┌────────────────────────┐ │                            │
│  │ Type a message...  [⏎] │ │                            │
│  └────────────────────────┘ │                            │
├─────────────────────────────┴────────────────────────────┤
│  Compare Mode: [Off ▾]                                   │
│  When ON: side-by-side chat with two different adapters  │
└──────────────────────────────────────────────────────────┘
```

**Features**:
- Load any downloaded model + any adapter checkpoint
- Streaming token generation (WebSocket)
- Token-by-token rendering in the chat UI
- Generation stats (tokens/s, total tokens)
- Compare mode: same prompt to two different adapters side-by-side
- Chat history export (JSON)
- Clear context button

**Streaming inference WebSocket**:

```
WS /ws/inference

Client → Server:
{"type": "generate", "messages": [...], "config": {"temperature": 0.7, ...}}

Server → Client (streaming):
{"type": "token", "text": "Lo"}
{"type": "token", "text": "RA"}
{"type": "token", "text": " stands"}
...
{"type": "done", "stats": {"num_tokens": 128, "tokens_per_second": 42.3}}
```

---

## §4. Contract Preservation

V1 makes **additive extensions only**. All v0 frozen contracts remain unchanged.

| Contract | v0 Spec | V1 Change |
|----------|---------|-----------|
| Config schema | V0_DESIGN_FREEZE.md §2.1 | New optional fields only (`quantization`, `packing`, `gradient_checkpointing`) |
| Batch contract | `(B, T)` + `(B, 2)` | Packed batches add `segment_ids: (B, T)` as optional third element |
| Checkpoint format | 3 files (adapters, optimizer, state.json) | Unchanged |
| Run directory layout | config.yaml, manifest.json, etc. | Unchanged |
| Data cache format | meta.json + shard safetensors | Unchanged |
| `state.json` schema_version | 1 | Remains 1 (no new fields in state) |

### Backward Compatibility Rules

1. Existing v0 configs (without new fields) continue to work unchanged.
2. Existing checkpoints can be resumed by V1.
3. Existing run directories are readable by Studio.
4. New configs with V1 fields are rejected by v0 (Pydantic `extra="forbid"` — this is correct behavior).
5. The `generate` command and Studio are purely additive (new CLI subcommands, new package).

---

## §5. Package Layout Changes

New files and directories in **bold**:

```
lmforge/
├── __init__.py                     # Updated: add generate() API
├── _version.py                     # Updated: "1.0.0"
├── config.py                       # Updated: new optional fields
├── manifest.py
│
├── data/
│   ├── formats.py
│   ├── preprocessing.py
│   ├── cache.py
│   ├── batching.py
│   └── **packing.py**              # NEW: sequence packing
│
├── adapters/
│   ├── targeting.py
│   └── lora.py
│
├── models/
│   ├── registry.py                 # Updated: add gemma
│   ├── resolve.py
│   ├── loader.py
│   ├── **quantize.py**             # NEW: quantization for QLoRA
│   ├── _base/
│   │   ├── args.py
│   │   ├── attention.py
│   │   ├── rope.py
│   │   └── activations.py
│   └── architectures/
│       ├── llama.py                # Updated: KV cache support
│       ├── qwen3.py                # Updated: KV cache support
│       ├── phi3.py                 # Updated: KV cache support
│       └── **gemma.py**            # NEW: Gemma 1/2/3
│
├── **inference/**                  # NEW: inference engine
│   ├── **__init__.py**
│   ├── **engine.py**               # Generation loop
│   ├── **sampling.py**             # Temperature, top-p sampling
│   └── **cache.py**                # KV cache implementation
│
├── trainer/
│   ├── trainer.py                  # Updated: gradient checkpointing, packing
│   ├── state.py
│   ├── callbacks.py
│   ├── checkpoint.py
│   └── optimizer.py
│
├── logging/
│   └── metrics.py
│
├── cli/
│   ├── main.py                     # Updated: add generate + studio commands
│   ├── prepare_cmd.py
│   ├── train_cmd.py                # Updated: wire --resume
│   ├── **generate_cmd.py**         # NEW: lmforge generate
│   └── **studio_cmd.py**           # NEW: lmforge studio
│
└── **studio/**                     # NEW: LMForge Studio backend
    ├── **__init__.py**
    ├── **server.py**               # FastAPI app + WebSocket hub
    ├── **api/**
    │   ├── **runs.py**             # Experiment endpoints
    │   ├── **models.py**           # Model management endpoints
    │   ├── **datasets.py**         # Dataset endpoints
    │   ├── **training.py**         # Training control endpoints
    │   └── **inference.py**        # Generation endpoints
    ├── **services/**
    │   ├── **run_service.py**      # Run discovery + metrics parsing
    │   ├── **model_service.py**    # Model discovery from HF cache
    │   ├── **dataset_service.py**  # Dataset cache scanning
    │   ├── **training_service.py** # Training process management
    │   └── **metrics_watcher.py**  # Real-time metrics file watcher
    └── **frontend/**               # Pre-built React SPA
        ├── **index.html**
        ├── **assets/**
        │   ├── **index-[hash].js**
        │   └── **index-[hash].css**
        └── **manifest.json**       # PWA manifest
```

#### Frontend Source (Development)

```
studio-frontend/                    # Separate directory, NOT in lmforge/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.ts
├── index.html
├── public/
│   ├── favicon.svg
│   └── manifest.json
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── api/                        # API client
    │   ├── client.ts
    │   └── types.ts
    ├── hooks/                      # Custom React hooks
    │   ├── useRuns.ts
    │   ├── useModels.ts
    │   ├── useWebSocket.ts
    │   └── useMetrics.ts
    ├── pages/
    │   ├── Dashboard.tsx
    │   ├── Experiments.tsx
    │   ├── RunDetail.tsx
    │   ├── RunCompare.tsx
    │   ├── Models.tsx
    │   ├── Datasets.tsx
    │   ├── Playground.tsx
    │   └── Settings.tsx
    ├── components/
    │   ├── layout/
    │   │   ├── Sidebar.tsx
    │   │   └── PageLayout.tsx
    │   ├── charts/
    │   │   ├── LossChart.tsx
    │   │   ├── ThroughputChart.tsx
    │   │   └── MemoryChart.tsx
    │   ├── experiments/
    │   │   ├── RunCard.tsx
    │   │   ├── RunTable.tsx
    │   │   └── ConfigDiff.tsx
    │   ├── playground/
    │   │   ├── ChatMessage.tsx
    │   │   ├── ChatInput.tsx
    │   │   └── GenerationConfig.tsx
    │   └── shared/
    │       ├── StatusBadge.tsx
    │       ├── MetricCard.tsx
    │       └── EmptyState.tsx
    └── lib/
        ├── utils.ts
        └── constants.ts
```

The frontend is built with `npm run build` and the output is copied to `lmforge/studio/frontend/` before publishing to PyPI. Users never need Node.js.

---

## §6. Implementation Plan

### Phase 1: Foundation (Week 1-2)

| Task | Files | Effort |
|------|-------|--------|
| Fix resume from checkpoint | `cli/train_cmd.py`, `__init__.py`, `trainer/trainer.py` | 1 day |
| Gemma architecture | `models/architectures/gemma.py`, `models/registry.py` | 1-2 days |
| KV cache for all architectures | `inference/cache.py`, all architecture files | 2 days |
| Inference engine (sampling, generation loop) | `inference/engine.py`, `inference/sampling.py` | 2 days |
| `lmforge generate` CLI | `cli/generate_cmd.py`, `cli/main.py` | 1 day |
| `generate()` library API | `__init__.py` | 1 day |
| Tests for inference | `tests/test_inference.py` | 1 day |

### Phase 2: Performance (Week 3-4)

| Task | Files | Effort |
|------|-------|--------|
| QLoRA quantization | `models/quantize.py`, `config.py` | 2-3 days |
| Gradient checkpointing | `trainer/trainer.py`, `config.py` | 1-2 days |
| Sequence packing | `data/packing.py`, `trainer/trainer.py` | 3-4 days |
| Tests for performance features | `tests/test_qlora.py`, `tests/test_packing.py` | 2 days |

### Phase 3: Studio Backend (Week 5-6)

| Task | Files | Effort |
|------|-------|--------|
| FastAPI server + static file serving | `studio/server.py` | 1 day |
| Run service (discovery, metrics parsing) | `studio/services/run_service.py` | 2 days |
| Model service (HF cache scanning) | `studio/services/model_service.py` | 1 day |
| Dataset service (cache scanning) | `studio/services/dataset_service.py` | 1 day |
| REST API routes | `studio/api/*.py` | 2 days |
| WebSocket hub (training + inference) | `studio/server.py`, `studio/services/metrics_watcher.py` | 2 days |
| Training service (start/stop/resume) | `studio/services/training_service.py` | 2 days |
| `lmforge studio` CLI command | `cli/studio_cmd.py`, `cli/main.py` | 1 day |

### Phase 4: Studio Frontend (Week 7-9)

| Task | Files | Effort |
|------|-------|--------|
| Project scaffolding (Vite + React + Tailwind + shadcn) | `studio-frontend/` | 1 day |
| Layout (sidebar, routing) | `components/layout/*`, `App.tsx` | 1 day |
| Dashboard page | `pages/Dashboard.tsx` | 1 day |
| Experiments list page | `pages/Experiments.tsx`, `components/experiments/*` | 2 days |
| Run detail page (loss charts, config, checkpoints) | `pages/RunDetail.tsx`, `components/charts/*` | 3 days |
| Run comparison page | `pages/RunCompare.tsx` | 2 days |
| Models page | `pages/Models.tsx` | 1 day |
| Datasets page | `pages/Datasets.tsx` | 1 day |
| Playground page (chat + streaming) | `pages/Playground.tsx`, `components/playground/*` | 3 days |
| Settings page | `pages/Settings.tsx` | 0.5 day |
| Build pipeline (Vite → lmforge/studio/frontend/) | `vite.config.ts`, build script | 0.5 day |
| PWA manifest + service worker | `public/manifest.json` | 0.5 day |

### Phase 5: Integration & Polish (Week 10)

| Task | Files | Effort |
|------|-------|--------|
| End-to-end integration tests | `tests/test_studio.py`, `tests/test_e2e_v1.py` | 2 days |
| Error handling and edge cases | All files | 2 days |
| Documentation update | `CLAUDE.md`, `README.md` | 1 day |

### Total Estimated Effort: ~10 weeks

---

## §7. What V1 Does NOT Do

| Feature | Reason |
|---------|--------|
| Native desktop app (Tauri/Electron) | Web app first; Tauri is a v2 consideration |
| Cloud sync / accounts / telemetry | Local-first philosophy |
| Background daemon | Studio is on-demand, not always-running |
| Database (SQLite, Postgres) | Filesystem is the database |
| Multi-user collaboration | Single-user tool |
| Model conversion (HF ↔ MLX) | Delegated to external tools |
| DPO / RLHF training | Different paradigm, v2+ |
| DoRA / IA3 adapters | Stretch goal, not blocking |
| MoE models | High complexity, low priority for Apple Silicon |
| Beam search / speculative decoding | Greedy + sampling sufficient for testing |
| OpenAI-compatible serving API | Serving is a separate concern |
| Distributed training | Apple Silicon is single-GPU |
| Model upload to HuggingFace | Users can use `huggingface-cli` directly |
| Notebook / Jupyter integration | Out of scope |

---

## §8. Design Decisions Summary

| # | Decision | Choice | Alternatives Considered | Rationale |
|---|----------|--------|------------------------|-----------|
| 1 | UI technology | Web app (FastAPI + React) | Electron, Tauri, Streamlit | Zero extra dependencies, proven ML pattern, Tauri door open for v2 |
| 2 | Studio naming | LMForge Studio | Lab, Workbench, Forge, Console | Professional, creative, matches "building something" |
| 3 | Frontend framework | React + Vite + TypeScript | Svelte, Vue, Streamlit | Largest ecosystem, most components, widest hiring pool |
| 4 | Real-time protocol | WebSocket | SSE, polling | Bidirectional (user can stop/pause), streaming inference |
| 5 | Data storage | Filesystem (no DB) | SQLite, Redis | Reads existing LMForge artifacts directly, no migration |
| 6 | Inference scope | Greedy + top-p sampling | Beam search, speculative | Sufficient for testing fine-tunes, minimal complexity |
| 7 | KV cache | Per-architecture injection | Shared cache module | Each architecture has specific attention patterns |
| 8 | Quantization | MLX native `nn.quantize()` | Custom GPTQ/AWQ | MLX handles all quantization natively, battle-tested |
| 9 | Packing attention | Block-diagonal mask via segment_ids | Separate attention kernel | Uses existing attention mechanism, no custom Metal |
| 10 | Gradient checkpointing | `mx.checkpoint()` per layer | Manual activation clearing | MLX provides this natively, one-line per layer |
| 11 | Studio as optional dep | `pip install lmforge[studio]` | Always bundled | Keeps core package minimal, no FastAPI for CLI-only users |
| 12 | Frontend distribution | Pre-built, shipped in Python package | npm install at runtime | Users never need Node.js installed |
| 13 | Resume data skip | Re-iterate and skip N batches | Save iterator state | Tier-1 resume (per V0_DESIGN_FREEZE.md §3), simple and correct enough |

---

## Appendix A: CLI Command Summary (V1)

```
lmforge prepare   --data FILE --model MODEL [--output DIR] [--max-seq-length N] [--no-mask-prompt]
lmforge train     --config FILE [--resume CHECKPOINT_DIR]
lmforge generate  --model MODEL [--adapter DIR] [--prompt TEXT] [--temperature F] [--top-p F] [--max-tokens N]
lmforge studio    [--port N] [--host HOST] [--no-browser] [--background]
```

## Appendix B: Dependency Changes

```toml
[project]
dependencies = [
    "mlx>=0.18.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "numpy>=1.24.0",
    "transformers>=4.35.0",
    "safetensors>=0.4.0",
    "huggingface-hub>=0.20.0",    # Already used by M7
]

[project.optional-dependencies]
studio = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "websockets>=12.0",
]
dev = [
    "pytest>=7.0",
    "pytest-timeout>=2.0",
    "pytest-asyncio>=0.23.0",     # NEW: for Studio API tests
    "httpx>=0.25.0",              # NEW: for FastAPI test client
]
wandb = ["wandb>=0.16.0"]
```

## Appendix C: Studio Settings File

```yaml
# ~/.lmforge/studio.yaml
port: 8741
host: "127.0.0.1"
theme: "dark"
run_dir: "~/.lmforge/runs"
auto_open_browser: true
```
