# Plan: M40–M44 — Inference-Grade Local Serving for Apple Silicon

## Context

MLX Forge v0.7.1 has a training pipeline that no competitor can match. But the inference/serving layer has critical gaps:

1. **Dead wiring** — Speculative decoding, prompt cache, CacheManager, RequestQueue, and 4 CLI sampling params are all implemented but never connected.
2. **No model lifecycle** — No keep-alive, no auto-unload, no multi-model, no aliases. Ollama's `keep_alive: "5m"` is table stakes.
3. **No context management** — No rotating KV cache, no memory-aware loading. Long conversations OOM silently.
4. **Missing competitive features** — No thinking model support, no fill-in-middle, no real grammar constraints.
5. **No train-to-serve bridge** — The unique selling point (train + serve in one tool) has no integrated workflow.

**Build order**: M40 (wire) → M41 (lifecycle) → M42 (context) → M43 (features) → M44 (bridge)

---

## M40: Wire Everything + Server Essentials (~35 tests)

### Goal
Every implemented feature actually works end-to-end. Zero silent failures.

### 1. Wire CLI Generate Command

**File: `mlx_forge/cli/generate_cmd.py`** — Fix `run_generate()` and `_run_interactive()`

```python
# run_generate() — pass ALL M37 params to generate()
def run_generate(args) -> None:
    from mlx_forge.inference.engine import generate, load_for_inference

    model, tokenizer = load_for_inference(
        args.model,
        adapter_path=args.adapter,
        trust_remote_code=args.trust_remote_code,
    )

    if args.prompt:
        result = generate(
            model, tokenizer, prompt=args.prompt,
            temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k, min_p=args.min_p,                        # NEW
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            frequency_penalty=args.frequency_penalty,                    # NEW
            presence_penalty=args.presence_penalty,                      # NEW
            seed=args.seed,
        )
        _print_result(result)
    elif args.draft_model:
        _run_speculative(model, tokenizer, args)                         # NEW
    else:
        _run_interactive(model, tokenizer, args)
```

**New function: `_run_speculative()`** (~30 lines)
```python
def _run_speculative(model, tokenizer, args):
    """Single-shot generation with speculative decoding."""
    from mlx_forge.inference.engine import load_for_inference
    from mlx_forge.inference.speculative import speculative_generate_tokens

    draft_model, _ = load_for_inference(args.draft_model,
                                         trust_remote_code=args.trust_remote_code)
    prompt_tokens = tokenizer.encode(args.prompt or "")
    # ... generate and print with accepted/rejected stats
```

**Fix `_run_interactive()`** — pass M37 params to `generate_tokens()`:
```python
for token_id in generate_tokens(
    model, prompt_tokens, tokenizer,
    temperature=args.temperature, top_p=args.top_p,
    top_k=getattr(args, 'top_k', 0),                   # NEW
    min_p=getattr(args, 'min_p', 0.0),                  # NEW
    max_tokens=args.max_tokens,
    repetition_penalty=args.repetition_penalty,
    frequency_penalty=getattr(args, 'frequency_penalty', 0.0),  # NEW
    presence_penalty=getattr(args, 'presence_penalty', 0.0),    # NEW
):
```

### 2. Wire CacheManager into Server

**File: `mlx_forge/inference/engine.py`** — Add `cache` and `all_token_history` params to `generate_steps()`

```python
def generate_steps(
    model, prompt_tokens, tokenizer, *,
    cache=None,                     # NEW: pass existing KV cache
    all_token_history=None,         # NEW: full history for repetition penalty
    temperature=0.7, top_p=0.9, top_k=0, min_p=0.0,
    max_tokens=512, repetition_penalty=1.0,
    frequency_penalty=0.0, presence_penalty=0.0,
    seed=None, logprobs=False, top_logprobs=5,
) -> Iterator[StepResult]:
    """If cache is provided, prompt_tokens are ONLY the new tokens to prefill.
    The cache already contains earlier context. all_token_history provides
    the full token sequence for repetition/frequency penalty tracking."""

    if seed is not None:
        mx.random.seed(seed)

    if cache is None:
        cache_max_size = len(prompt_tokens) + max_tokens
        cache = _make_model_cache(model, max_size=cache_max_size)

    # Prefill whatever tokens are given
    tokens = mx.array(prompt_tokens)[None]
    model_logits = model(tokens, cache=cache)

    # For penalties: track full history (not just new tokens)
    generated = list(all_token_history or prompt_tokens)
    # ... rest unchanged
```

**File: `mlx_forge/serving/app.py`** — Create CacheManager at startup, expose via state

```python
from mlx_forge.serving.cache_manager import CacheManager

def create_serving_app(
    model: str | None = None,
    adapter: str | None = None,
    draft_model: str | None = None,          # NEW
    max_conversations: int = 16,             # NEW
    cache_ttl: float = 600,                  # NEW
) -> FastAPI:
    app = FastAPI(title="MLX Forge Serving", version="1.0.0")

    # Create shared cache manager
    cache_mgr = CacheManager(
        max_conversations=max_conversations,
        ttl_seconds=cache_ttl,
    )
    app.state.cache_manager = cache_mgr

    app.include_router(router)

    if model:
        @app.on_event("startup")
        async def preload_model():
            mgr = ModelManager()
            mgr.load(model, adapter=adapter)
            mgr.snapshot_base_weights()       # NEW: always snapshot
            if draft_model:
                mgr.load_draft(draft_model)   # NEW
            set_manager(mgr)

    return app
```

**File: `mlx_forge/serving/routes.py`** — Wire CacheManager into chat endpoint

Add `conversation_id` extension field to `ChatCompletionRequest`:
```python
class ChatCompletionRequest(BaseModel):
    # ... existing ...
    conversation_id: str | None = None   # NEW: extension for multi-turn cache
```

Wire into `chat_completions()`:
```python
@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    # ... existing setup ...

    cache_mgr = getattr(req.app.state, 'cache_manager', None)
    conv_id = request.conversation_id

    if cache_mgr and conv_id:
        cache, new_tokens = cache_mgr.get_or_create(conv_id, prompt_tokens, mgr.model)
        all_history = list(prompt_tokens)  # full history for penalties
    else:
        cache, new_tokens, all_history = None, prompt_tokens, None

    for step in generate_steps(
        mgr.model, new_tokens, mgr.tokenizer,
        cache=cache,
        all_token_history=all_history,
        # ... all other params ...
    ):
        # ... existing generation loop ...

    # After generation, update cache
    if cache_mgr and conv_id and cache is not None:
        all_tokens = list(prompt_tokens) + generated_ids
        cache_mgr.update(conv_id, cache, all_tokens)
```

### 3. Health Endpoint + Timing Metadata

**File: `mlx_forge/serving/routes.py`** — New endpoint

```python
@router.get("/health")
async def health():
    """Health check — Ollama-compatible."""
    mgr = get_manager()
    return {
        "status": "ok",
        "model_loaded": mgr.is_loaded,
        "model_id": mgr.model_id,
    }

@router.get("/v1/models/{model_id}")
async def get_model_info(model_id: str):
    """Model metadata — context length, quantization, capabilities."""
    mgr = get_manager()
    # ... return model info from config.json
```

**File: `mlx_forge/serving/openai_types.py`** — Add Ollama-compatible timing

```python
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Timing extensions (Ollama-compatible, in milliseconds)
    ttft_ms: float | None = None
    prompt_eval_duration_ms: float | None = None    # NEW
    eval_duration_ms: float | None = None           # NEW
    decode_tokens_per_sec: float | None = None
```

### 4. Wire Speculative Decoding into Server

**File: `mlx_forge/serving/routes.py`** — Branch on draft model presence

```python
# In chat_completions() and completions():
if mgr.has_draft:
    from mlx_forge.inference.speculative import speculative_generate_tokens
    for token_id, from_draft in speculative_generate_tokens(
        mgr.model, mgr.draft_model, prompt_tokens, mgr.tokenizer,
        num_draft=5, temperature=request.temperature, ...
    ):
        # ... same token processing ...
else:
    for step in generate_steps(...):
        # ... existing path ...
```

**File: `mlx_forge/cli/main.py`** — Add `--draft-model` to serve command (already has it, just needs wiring)

**File: `mlx_forge/cli/serve_cmd.py`** — Pass draft_model to app factory

```python
app = create_serving_app(
    model=args.model,
    adapter=getattr(args, "adapter", None),
    draft_model=getattr(args, "draft_model", None),   # NEW
)
```

### 5. Wire Prompt Cache into CLI

**File: `mlx_forge/cli/generate_cmd.py`** — Read `args.prompt_cache`

```python
# In _run_interactive(), before generation loop:
if args.prompt_cache and Path(args.prompt_cache).exists():
    from mlx_forge.inference.prompt_cache import load_prompt_cache, apply_prompt_cache
    tensors, metadata = load_prompt_cache(args.prompt_cache)
    # ... apply to cache ...

# After first generation, optionally save:
if args.prompt_cache and not Path(args.prompt_cache).exists():
    from mlx_forge.inference.prompt_cache import save_prompt_cache
    # ... save cache state ...
```

### M40 Tests: `tests/test_m40_wiring.py` (~35 tests)

```
TestCLIParamPassthrough (6):
  top_k reaches generate, min_p reaches generate, freq_penalty reaches generate,
  pres_penalty reaches generate, backward compat (old params only), interactive mode passes params

TestSpeculativeWiring (4):
  draft_model loads and generates, server branch when has_draft,
  fallback when no draft, draft_model in serve_cmd

TestPromptCacheWiring (4):
  save cache file, load cache file, apply to fresh cache, cli flag creates file

TestCacheManagerWiring (8):
  conversation_id enables caching, no conv_id skips cache, prefix reuse in server,
  divergent prompt resets, cache survives across requests, eviction under pressure,
  cache stats endpoint, generate_steps accepts external cache

TestHealthEndpoint (4):
  health returns ok, health shows model loaded, health shows model_id,
  health when no model returns not loaded

TestTimingMetadata (4):
  ttft_ms in chat response, eval_duration in response, prompt_eval_duration in response,
  streaming still works with timing

TestModelInfo (3):
  model info endpoint, unknown model 404, model capabilities list

TestBackwardCompat (2):
  existing tests still pass, no conversation_id works as before
```

---

## M41: Model Lifecycle Management (~35 tests)

### Goal
`mlx-forge serve` feels as effortless as Ollama. Models warm up, stay alive, auto-unload.

### 1. ModelPool — Multi-Model Manager

**New file: `mlx_forge/serving/model_pool.py`** (~200 lines)

```python
@dataclass
class ManagedModel:
    """A model with lifecycle metadata."""
    manager: ModelManager
    model_id: str
    loaded_at: float
    last_access: float
    keep_alive: float          # seconds, -1 = forever, 0 = immediate unload
    memory_bytes: int = 0

class ModelPool:
    """Manages multiple models with TTL-based lifecycle.

    Like Ollama: models load on demand, stay alive for keep_alive seconds,
    then auto-unload to free memory. LRU eviction when max_models exceeded.
    """

    def __init__(self, max_models: int = 1, default_keep_alive: float = 300):
        self._models: dict[str, ManagedModel] = {}
        self._aliases: dict[str, str] = {}       # alias → model_id
        self._max_models = max_models
        self._default_keep_alive = default_keep_alive

    def get(self, model_id: str, keep_alive: float | None = None) -> ManagedModel:
        """Get a loaded model, loading it if necessary.
        Resolves aliases. Evicts idle models if needed."""

    def unload(self, model_id: str) -> bool:
        """Explicitly unload a model."""

    def tick(self) -> list[str]:
        """Evict expired models. Call periodically (e.g., every 30s).
        Returns list of evicted model IDs."""

    def status(self) -> list[dict]:
        """Return status of all loaded models (like Ollama /api/ps)."""

    def resolve_alias(self, name: str) -> str:
        """Resolve alias to model_id."""

    def add_alias(self, alias: str, model_id: str): ...
    def load_aliases(self, path: Path): ...
    def save_aliases(self, path: Path): ...
```

### 2. Alias System

**File: `~/.mlxforge/aliases.json`** — User-maintained aliases

```json
{
  "chat": "Qwen/Qwen3-0.6B",
  "code": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "my-assistant": "forge:my-assistant"
}
```

Loaded at startup. Users can also set via CLI:
```bash
mlx-forge alias set chat Qwen/Qwen3-0.6B
mlx-forge alias list
mlx-forge alias remove chat
```

**File: `mlx_forge/cli/main.py`** — New `alias` subcommand

### 3. Keep-Alive Per-Request

**File: `mlx_forge/serving/openai_types.py`** — Add keep_alive to requests

```python
class ChatCompletionRequest(BaseModel):
    # ... existing ...
    keep_alive: str | int | None = None    # NEW: "5m", "1h", 300, -1, 0
```

**File: `mlx_forge/serving/model_pool.py`** — Parse keep_alive strings

```python
def parse_keep_alive(value: str | int | None, default: float) -> float:
    """Parse Ollama-style keep_alive values.
    "5m" → 300, "1h" → 3600, -1 → inf, 0 → 0, None → default."""
```

### 4. Background Eviction

**File: `mlx_forge/serving/app.py`** — Periodic tick task

```python
@app.on_event("startup")
async def start_eviction_loop():
    async def _evict_loop():
        while True:
            await asyncio.sleep(30)
            pool.tick()
    asyncio.create_task(_evict_loop())
```

### 5. Server Status Endpoints

**File: `mlx_forge/serving/routes.py`** — New endpoints

```python
@router.get("/v1/models/ps")
async def running_models():
    """List running models with memory usage and TTL (Ollama /api/ps equivalent)."""

@router.post("/v1/models/load")
async def preload_model(request: ModelLoadRequest):
    """Pre-warm a model without generating (empty request)."""

@router.post("/v1/models/unload")
async def unload_model(request: ModelUnloadRequest):
    """Explicitly unload a model."""
```

### 6. CLI Flags

**File: `mlx_forge/cli/main.py`** — New serve flags

```python
serve_parser.add_argument("--max-models", type=int, default=1)
serve_parser.add_argument("--keep-alive", default="5m")
serve_parser.add_argument("--aliases", default="~/.mlxforge/aliases.json")
```

### M41 Tests: `tests/test_m41_lifecycle.py` (~35 tests)

```
TestModelPool (12):
  load on demand, reuse loaded, LRU eviction, max models enforced,
  unload explicit, unload frees memory, concurrent get same model,
  status returns all loaded, empty pool status, load unknown model error,
  memory tracking, reload after unload

TestKeepAlive (8):
  parse "5m" → 300, parse "1h" → 3600, parse -1 → inf, parse 0 → 0,
  parse int seconds, default when None, per-request override,
  tick evicts expired

TestAliases (6):
  resolve alias, unknown alias passthrough, add alias, remove alias,
  load from file, save to file

TestServerEndpoints (5):
  /v1/models/ps returns loaded, /v1/models/load warms model,
  /v1/models/unload frees, preload via empty messages, keep_alive in request

TestEvictionLoop (4):
  background tick runs, expired models evicted, pinned models survive,
  eviction under memory pressure
```

---

## M42: Context & Memory Management (~25 tests)

### Goal
Handle real-world conversation lengths on consumer Macs without OOM.

### 1. Rotating KV Cache

**New file: `mlx_forge/inference/rotating_cache.py`** (~80 lines)

```python
class RotatingKVCache:
    """Fixed-size KV cache with sliding window eviction.

    When the cache fills beyond max_size, keeps the first num_keep tokens
    (system prompt) and the most recent tokens, discarding the middle.
    This matches Ollama's behavior with num_keep + sliding window.
    """

    def __init__(self, max_size: int, num_keep: int = 0):
        self.max_size = max_size
        self.num_keep = num_keep
        # ... same interface as KVCache: offset, update_and_fetch(), trim()

    def update_and_fetch(self, keys, values):
        """If offset + new_len > max_size, rotate: keep first num_keep
        tokens + most recent tokens that fit."""
```

**File: `mlx_forge/inference/cache.py`** — Add `make_rotating_cache()` factory

```python
def make_rotating_cache(
    num_layers: int, *, max_size: int, num_keep: int = 0
) -> list[RotatingKVCache]:
    """Create rotating caches for context overflow handling."""
```

### 2. Inference Memory Estimation

**File: `mlx_forge/models/memory.py`** — Add inference-specific estimation

```python
@dataclass
class InferenceMemoryEstimate:
    """Memory breakdown for inference (not training)."""
    model_weights_gb: float
    kv_cache_gb: float         # for given context_length
    overhead_gb: float = 0.3
    total_gb: float = 0.0
    fits: bool = True
    max_context_that_fits: int = 0   # largest context that fits in memory

def estimate_inference_memory(
    model_id: str,
    *,
    context_length: int = 4096,
    quantization_bits: int | None = None,
    kv_quantization_bits: int = 16,    # future: 8 for q8_0 KV
    hardware: HardwareProfile | None = None,
) -> InferenceMemoryEstimate:
    """Estimate inference memory and compute max safe context length."""
```

### 3. Memory-Aware Model Loading

**File: `mlx_forge/serving/model_pool.py`** — Check memory before loading

```python
def get(self, model_id, keep_alive=None) -> ManagedModel:
    # ... existing ...
    # NEW: estimate memory, warn or auto-reduce context if needed
    estimate = estimate_inference_memory(model_id)
    if not estimate.fits:
        # Try evicting idle models first
        self._evict_until_fits(estimate.total_gb)
        if still not fits:
            # Auto-reduce context
            safe_ctx = estimate.max_context_that_fits
            logger.warning(f"Reducing context to {safe_ctx} to fit in memory")
```

### 4. Wire into Server

**File: `mlx_forge/cli/main.py`** — New serve flags

```python
serve_parser.add_argument("--context-length", type=int, default=4096)
serve_parser.add_argument("--num-keep", type=int, default=0,
    help="Tokens to always keep at start of context (system prompt)")
```

**File: `mlx_forge/serving/openai_types.py`** — Context options

```python
class ChatCompletionRequest(BaseModel):
    # ... existing ...
    num_ctx: int | None = None        # NEW: per-request context override
    num_keep: int | None = None       # NEW: tokens to preserve on rotation
```

### M42 Tests: `tests/test_m42_context.py` (~25 tests)

```
TestRotatingKVCache (10):
  basic update, rotation at max_size, num_keep preserved, recent tokens kept,
  trim works, reset works, offset tracking after rotation, small max_size,
  zero num_keep, compatibility with model forward

TestInferenceMemory (8):
  estimate fp16 model, estimate 4bit model, max_context_that_fits,
  kv_cache scales with context, known model profile, unknown model fallback,
  hardware detection, fits check

TestMemoryAwareLoading (4):
  warn when tight, auto-reduce context, evict idle before loading,
  refuse when impossible

TestContextOverflow (3):
  long conversation triggers rotation, system prompt preserved,
  generation quality maintained after rotation
```

---

## M43: Competitive Features (~30 tests)

### Goal
Feature parity with Ollama on the features that matter for Apple Silicon users.

### 1. Thinking Model Support

**New file: `mlx_forge/inference/thinking.py`** (~60 lines)

```python
@dataclass
class ThinkingResult:
    """Parsed thinking + response from a reasoning model."""
    thinking: str        # chain-of-thought content
    response: str        # final answer

class ThinkingParser:
    """Parse thinking model output (DeepSeek-R1, Qwen3, QwQ).

    Detects and separates <think>...</think> blocks from response."""

    PATTERNS = [
        (re.compile(r"<think>(.*?)</think>", re.DOTALL), "think_tags"),
        (re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL), "thinking_tags"),
    ]

    def parse(self, text: str) -> ThinkingResult: ...
    def is_thinking_model(self, model_config: dict) -> bool: ...
```

**File: `mlx_forge/serving/openai_types.py`** — Add think param and response field

```python
class ChatCompletionRequest(BaseModel):
    # ... existing ...
    think: bool = False              # NEW: enable extended thinking

class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCallMessage] | None = None
    thinking: str | None = None      # NEW: thinking content
```

**File: `mlx_forge/serving/routes.py`** — Parse thinking from output

```python
# After generation completes:
if request.think:
    from mlx_forge.inference.thinking import ThinkingParser
    parsed = ThinkingParser().parse(generated_text)
    message = ChatMessage(
        role="assistant",
        content=parsed.response,
        thinking=parsed.thinking,
    )
```

### 2. Fill-in-Middle (Code Completion)

**New file: `mlx_forge/inference/fim.py`** (~50 lines)

```python
# FIM template patterns for different model families
FIM_TEMPLATES = {
    "codellama": {"prefix": "<PRE>", "suffix": "<SUF>", "middle": "<MID>"},
    "deepseek":  {"prefix": "<｜fim▁begin｜>", "suffix": "<｜fim▁hole｜>", "middle": "<｜fim▁end｜>"},
    "starcoder": {"prefix": "<fim_prefix>", "suffix": "<fim_suffix>", "middle": "<fim_middle>"},
    "qwen":      {"prefix": "<|fim_prefix|>", "suffix": "<|fim_suffix|>", "middle": "<|fim_middle|>"},
}

def build_fim_prompt(prefix: str, suffix: str, model_type: str) -> str:
    """Build a fill-in-middle prompt for the given model family."""

def detect_fim_support(config: dict) -> str | None:
    """Check if model supports FIM and return template name."""
```

**File: `mlx_forge/serving/openai_types.py`** — Add suffix to CompletionRequest

```python
class CompletionRequest(BaseModel):
    # ... existing ...
    suffix: str | None = None        # NEW: fill-in-middle suffix
```

**File: `mlx_forge/serving/routes.py`** — Handle suffix in completions

```python
# In completions():
if request.suffix:
    from mlx_forge.inference.fim import build_fim_prompt, detect_fim_support
    fim_type = detect_fim_support(model_config)
    if fim_type:
        prompt = build_fim_prompt(request.prompt, request.suffix, fim_type)
        prompt_tokens = mgr.tokenizer.encode(prompt)
```

### 3. Grammar-Constrained JSON Generation

**File: `mlx_forge/inference/constrained.py`** — Add logit processor for JSON

```python
class JSONLogitProcessor:
    """Lightweight FSM that biases logits toward valid JSON during generation.

    Tracks JSON structural state (inside string, nesting depth, expected next
    token class) and applies logit bias to prevent structural violations.
    Not as robust as outlines, but zero-dependency and handles 95% of cases.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._state = "EXPECT_VALUE"   # FSM state
        self._depth = 0
        self._in_string = False
        self._escape_next = False
        # Pre-compute token → character mapping for structural tokens
        self._build_token_map()

    def process(self, logits: mx.array, generated_text: str) -> mx.array:
        """Apply logit bias based on current JSON state.
        Returns modified logits."""
        self._update_state(generated_text)
        # Boost tokens that produce valid JSON continuations
        # Suppress tokens that would break JSON structure
        ...

    def _update_state(self, text: str): ...
    def _build_token_map(self): ...
```

**Integration into `generate_steps()`** — Add optional `logit_processor` callback

```python
def generate_steps(
    model, prompt_tokens, tokenizer, *,
    logit_processor=None,            # NEW: callable(logits, text) -> logits
    ...
):
    # ... in sampling loop:
    if logit_processor is not None:
        last_logits = logit_processor(last_logits, current_text)
    next_token = sample_next_token(last_logits, ...)
```

### 4. Tokenize / Detokenize Endpoints

**File: `mlx_forge/serving/routes.py`** — Utility endpoints

```python
class TokenizeRequest(BaseModel):
    model: str
    text: str

class DetokenizeRequest(BaseModel):
    model: str
    tokens: list[int]

@router.post("/v1/tokenize")
async def tokenize(request: TokenizeRequest):
    _ensure_model_loaded(request.model)
    mgr = get_manager()
    tokens = mgr.tokenizer.encode(request.text)
    return {"tokens": tokens, "count": len(tokens)}

@router.post("/v1/detokenize")
async def detokenize(request: DetokenizeRequest):
    _ensure_model_loaded(request.model)
    mgr = get_manager()
    text = mgr.tokenizer.decode(request.tokens)
    return {"text": text}
```

### M43 Tests: `tests/test_m43_features.py` (~30 tests)

```
TestThinkingParser (6):
  parse think tags, parse thinking tags, no thinking passthrough,
  nested content, empty thinking, is_thinking_model detection

TestThinkingAPI (3):
  think=true returns thinking field, think=false no thinking field,
  streaming with thinking

TestFIM (6):
  codellama template, deepseek template, starcoder template, qwen template,
  detect_fim_support, suffix in completion request

TestJSONLogitProcessor (8):
  basic JSON object, nested object, array, string escaping,
  numeric values, boolean/null, depth tracking, state transitions

TestLogitProcessorIntegration (3):
  response_format triggers processor, json_schema uses processor,
  no processor when not requested

TestTokenizeDetokenize (4):
  tokenize text, detokenize tokens, round-trip, count matches
```

---

## M44: Train-to-Serve Bridge (~25 tests)

### Goal
The thing nobody else can do. One-command from dataset to API endpoint.

### 1. Forge Files — Model Bundle Spec

**New file: `mlx_forge/forge.py`** (~100 lines)

```python
@dataclass
class ForgeSpec:
    """A bundled model definition: base + adapter + system prompt + params."""
    name: str
    base: str                        # HF ID or local path
    adapter: str | None = None       # run:run_id, path, or None
    system: str | None = None        # default system prompt
    parameters: dict = field(default_factory=dict)  # temperature, top_k, etc.

    @classmethod
    def from_yaml(cls, path: str | Path) -> ForgeSpec: ...

    @classmethod
    def from_run(cls, run_id: str) -> ForgeSpec:
        """Create a ForgeSpec from a training run.
        Reads config.yaml to find base model, uses best checkpoint."""

    def save(self, path: str | Path): ...

    def to_model_load_args(self) -> dict:
        """Convert to args for ModelManager.load()."""
```

**Storage**: `~/.mlxforge/forges/{name}.yaml`

```yaml
# Example: ~/.mlxforge/forges/my-assistant.yaml
name: my-assistant
base: Qwen/Qwen3-0.6B
adapter: "run:sft-2026-04-12"
system: |
  You are a helpful coding assistant specializing in Python.
  Always provide clear, concise answers with code examples.
parameters:
  temperature: 0.7
  top_k: 40
  min_p: 0.05
  context_length: 8192
  keep_alive: "10m"
```

### 2. Model Resolution Chain

**File: `mlx_forge/serving/model_pool.py`** — Extended resolution

```python
def _resolve_model_id(self, model_id: str) -> tuple[str, str | None, dict]:
    """Resolve model_id to (base_path, adapter_path, params).

    Resolution order:
    1. forge:{name} → ~/.mlxforge/forges/{name}.yaml
    2. Alias → aliases.json
    3. run:{id} → ~/.mlxforge/runs/{id}/ (base + best checkpoint)
    4. export:{id} → ~/.mlxforge/exports/{id}/
    5. HF repo ID → download
    6. Local path
    """
```

### 3. CLI Commands

**File: `mlx_forge/cli/main.py`** — New `forge` subcommand

```bash
# Create a forge from a training run
mlx-forge forge create my-assistant --from-run sft-2026-04-12

# Create from a Forgefile
mlx-forge forge create my-assistant -f Forgefile.yaml

# List forges
mlx-forge forge list

# Serve a forge
mlx-forge serve --model forge:my-assistant

# Delete a forge
mlx-forge forge delete my-assistant
```

**File: `mlx_forge/cli/main.py`** — New `train-and-serve` command

```bash
# Train, fuse, and start serving — one command
mlx-forge train-and-serve --config train.yaml --port 8000
```

**File: `mlx_forge/cli/train_and_serve_cmd.py`** (~50 lines)

```python
def run_train_and_serve(args):
    """Train → create forge → start server."""
    from mlx_forge.cli.train_cmd import run_train
    from mlx_forge.cli.serve_cmd import run_serve

    # 1. Train
    run_id = run_train(args)

    # 2. Create forge from run
    from mlx_forge.forge import ForgeSpec
    forge = ForgeSpec.from_run(run_id)
    forge.save(Path(f"~/.mlxforge/forges/{run_id}.yaml").expanduser())

    # 3. Start server
    args.model = f"forge:{run_id}"
    run_serve(args)
```

### 4. Adapter Switching by Run ID

**File: `mlx_forge/serving/routes.py`** — Enhanced adapter endpoint

```python
class AdapterLoadRequest(BaseModel):
    adapter_path: str | None = None
    run_id: str | None = None        # NEW: load by run ID

@router.post("/v1/adapters/load")
async def load_adapter(request: AdapterLoadRequest):
    if request.run_id:
        # Resolve run_id to adapter path
        adapter_path = _resolve_run_adapter(request.run_id)
    else:
        adapter_path = request.adapter_path
    # ... existing load logic
```

### M44 Tests: `tests/test_m44_bridge.py` (~25 tests)

```
TestForgeSpec (8):
  from_yaml, from_run, save, to_model_load_args, default params,
  adapter resolution (run: prefix), system prompt, parameters merge

TestForgeResolution (5):
  resolve forge:name, resolve run:id, resolve alias, resolve HF id,
  resolution priority order

TestForgeCLI (4):
  create from run, create from file, list forges, delete forge

TestTrainAndServe (4):
  end-to-end flow, creates forge, starts server, uses correct adapter

TestAdapterByRunId (4):
  load by run_id, run_id resolves to best checkpoint, unknown run_id error,
  switch between run_ids
```

---

## Summary

| Milestone | New Files | Modified Files | Tests | Key Deliverable |
|-----------|-----------|---------------|-------|-----------------|
| M40 | 0 | 6 (generate_cmd, engine, app, routes, types, serve_cmd) | ~35 | Zero dead code, health endpoint, cache wired |
| M41 | 1 (model_pool.py) | 4 (app, routes, types, main) | ~35 | Multi-model, keep-alive, aliases, /ps |
| M42 | 1 (rotating_cache.py) | 4 (cache, memory, routes, main) | ~25 | Context overflow handling, memory safety |
| M43 | 2 (thinking.py, fim.py) | 3 (constrained, engine, routes) | ~30 | Thinking models, FIM, JSON grammar |
| M44 | 2 (forge.py, train_and_serve_cmd.py) | 3 (model_pool, routes, main) | ~25 | Forge files, train-to-serve, run_id adapters |
| **Total** | **6 new** | **12 modified** | **~150** | Production inference on Apple Silicon |

## Build Dependencies

```
M40 ──→ M41 ──→ M42
  │       │
  │       └──→ M44 (needs ModelPool for forge resolution)
  │
  └──→ M43 (needs generate_steps logit_processor from M40)
```

M40 is prerequisite for everything. M41-M43 can partially parallelize after M40. M44 needs M41.

## Critical Design Decisions

1. **`generate_steps()` gets `cache` param, not a new function** — Minimal API change. Callers that don't pass cache get the same behavior. Callers that do (routes with CacheManager) only prefill delta tokens.

2. **`ModelPool` replaces `ModelManager` in server** — ModelManager stays as-is (loads one model). ModelPool wraps N ModelManagers. Routes call `pool.get(model_id)` instead of `_ensure_model_loaded()`.

3. **Rotating cache is a separate class, not a mode on KVCache** — KVCache's contract (pre-allocated slice assignment) is frozen. RotatingKVCache has different semantics (can discard tokens). Keeping them separate avoids breaking the batch contract.

4. **JSON logit processor is a callback, not baked into sampling** — `logit_processor: Callable[[mx.array, str], mx.array]` passed to `generate_steps()`. Keeps sampling.py pure. Allows future processors (outlines, custom grammars) without touching the core loop.

5. **ForgeSpec is YAML, not Dockerfile-like** — Consistency with training configs. No new DSL to learn. Can be generated programmatically from training runs.

6. **Thinking parsing is post-generation, not mid-stream** — Thinking tags are model-specific. Parsing mid-stream is fragile (partial tags). Post-generation parsing is robust and covers all formats.

## Verify After Each Milestone

```bash
# Tests
.venv/bin/python -m pytest tests/ -v
.venv/bin/ruff check mlx_forge/ tests/

# M40 quick check:
mlx-forge generate --model Qwen/Qwen3-0.6B --prompt "Hello" --top-k 50 --min-p 0.05
curl localhost:8000/health

# M41 quick check:
mlx-forge serve --model Qwen/Qwen3-0.6B --keep-alive 5m --max-models 2
curl localhost:8000/v1/models/ps

# M42 quick check:
mlx-forge serve --model Qwen/Qwen3-0.6B --context-length 32768 --num-keep 256

# M43 quick check:
curl localhost:8000/v1/chat/completions -d '{"model":"...","messages":[...],"think":true}'
curl localhost:8000/v1/completions -d '{"model":"...","prompt":"def hello(","suffix":"    return greeting"}'

# M44 quick check:
mlx-forge forge create my-bot --from-run my-training-run
mlx-forge serve --model forge:my-bot
```
