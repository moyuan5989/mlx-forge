# Changelog

All notable changes to MLX Forge will be documented in this file.

## [0.6.3] - 2026-03-23

### Architecture Expansion (M28)
- **25+ model architectures** (up from 8): Added Mixtral (MoE), DeepSeek V2/V3 (MoE+MLA), Cohere/Cohere2, Llama 4, Mamba/Mamba2 (SSM), Jamba (hybrid), Falcon H1, OLMo2, InternLM2, StarCoder2, GLM4, Granite, StableLM, OpenELM
- New base utilities: `SwitchGLU` (MoE routing), `MambaBlock` (SSM), `MultiLatentAttention` (MLA)
- `ArraysCache` for SSM/recurrent layer state
- GGUF weight mappings for Cohere, Mixtral, InternLM2, GLM4, StarCoder2

### New Training Methods (M29)
- **ORPO** — Odds Ratio Preference Optimization (no reference model needed)
- **KTO** — Kahneman-Tversky Optimization for unpaired preference data
- **SimPO** — Simple Preference Optimization with length normalization
- **GRPO fix** — Now uses frozen reference model weights instead of `stop_gradient` for correct KL computation
- New data format: KTO (`{"text": "...", "label": 1}`)

### GGUF Quantized Export (M30)
- `--quantize q4_0` produces 4x smaller GGUF files for Ollama/llama.cpp
- `--quantize q8_0` produces 2x smaller files with higher quality
- Block quantization in pure Python/NumPy (no external dependencies)

### Speculative Decoding (M31)
- `--draft-model` flag for 1.5-2x faster inference
- Draft-verify-accept/reject loop with automatic cache rewind
- `KVCache.trim()` for speculative rewind
- Prompt caching: save/load KV cache state to disk

### Streaming Data Pipeline (M33)
- `data.streaming: true` for datasets that don't fit in RAM
- `StreamingHFDataset` — wraps HF `IterableDataset` with shuffle buffer
- `StreamingJSONLDataset` — memory-efficient line-by-line JSONL reading
- `tokenize_single()` for on-the-fly tokenization

### Vision Model Support (M32 + M34)
- VLM inference via `mlx-vlm` integration (`pip install mlx-forge[vision]`)
- VLM LoRA fine-tuning with `VLMSFTTrainer` (batch_size=1, gradient accumulation)
- `VisionDataCollator` for image+text processing
- `--vision` flag on generate, serve, and train commands

### Memory Safety & Stability
- **Auto memory adjustment**: Detects hardware at runtime, auto-enables gradient checkpointing (>80% budget) and reduces batch_size (>90% budget) for ANY model on ANY Apple Silicon
- **DeltaNet float32 fix**: Per-chunk upcast in `gated_delta_chunkwise()` reduces peak memory 32x (was upcasting full sequence before loop)
- **Safe wired limit**: `mx.set_wired_limit()` reduced to 75% of max to prevent macOS freeze
- **Recurrent layer detection**: Memory estimator auto-applies 2x multiplier for DeltaNet/SSM/Mamba layers

### Studio Bug Fixes
- **Stop training button fixed**: `track_id` contained "/" from HF model names breaking URL routing → sanitized to use "--"
- **Status detection fixed**: Replaced 60s mtime timeout with heartbeat file + 300s window + subprocess liveness check
- **run_id tracking fixed**: Queue jobs now report real run_id immediately via `on_run_id` callback instead of after process exit
- **Duplicate job prevention**: Queue discards stale jobs on restart; rejects duplicate model+dataset submissions
- **Queue subprocess kill**: `cancel()` now sends SIGINT to subprocess before updating job status
- **Shared TrainingService**: Queue uses shared instance so `getActiveTraining()` sees queue-spawned processes

### Tests
- 180 new tests across 7 test files (M28-M34)
- Total: 897+ tests, all passing on Python 3.10/3.11/3.12

## [0.5.0] - 2026-03-17

### Features
- **DoRA Support** (M23): Weight-Decomposed Low-Rank Adaptation via `adapter.method: "dora"` — adds learned magnitude vector on top of LoRA for better fine-tuning quality
- **OpenAI-Compatible API Server** (M20): `mlx-forge serve` starts an OpenAI-compatible server with `/v1/chat/completions`, `/v1/completions`, and `/v1/models` endpoints — works with Cursor, Continue.dev, Open WebUI, LangChain
- **HuggingFace Hub Upload** (M21): `--push-to-hub` flag on export command, auto-generated model cards with YAML frontmatter, "Push to Hub" button in Studio UI
- **HuggingFace Datasets Integration** (M24): `data.hf_dataset` config field to load any of HuggingFace's 200k+ datasets, `mlx-forge data hf-import` CLI command, auto-detection of 7 dataset formats (Alpaca, ShareGPT, OpenAI messages, text, preference, Q&A)
- **GGUF Export** (M22): `--format gguf` on export command for Ollama/llama.cpp deployment, supports Llama, Mistral, Qwen2, and Phi-3 architectures
- **Full Fine-Tuning** (M27): `adapter.method: "full"` for training all model parameters, memory warning printed, incompatible with quantization by design
- **GRPO Training** (M26): Group Relative Policy Optimization via `training_type: "grpo"` with built-in reward functions (length, rule-based, external API), clipped surrogate + KL penalty
- **MkDocs Documentation** (M25): Full documentation site with Material theme, 7 pages covering installation, quick start, CLI/config reference, Studio guide, OpenAI API, and architecture guide

### Tests
- Added 145 new tests across 7 test files (717 total, up from 572)

## [0.3.0] - 2026-03-09

### Features
- **Adapter Export**: `mlx-forge export` CLI command to merge LoRA adapters into base model and save as standalone safetensors
- **Export from Studio**: "Export Merged Model" button on run detail page
- **Training Controls**: Stop training button in Studio (Dashboard and Run Detail)
- **Queue Persistence**: Job queue survives server restarts via `~/.mlxforge/queue.json`
- **Error Boundary**: React error boundary wraps all pages — no more white screens

### Security
- Input validation on all API endpoints to prevent path traversal attacks
- SPA fallback rejects `..` and null-byte paths
- Adapter path validation ensures paths stay within `~/.mlxforge/`

### Bug Fixes
- Top-p sampling: fallback to top-1 when top_p is very small (prevents NaN)
- Temperature floor: `temperature=0` no longer causes division by zero
- Repetition penalty: out-of-range token IDs are clamped to vocab size
- Playground: API errors now show as dismissible banner instead of chat messages

### Docs
- Fixed broken repository URLs in CONTRIBUTING.md
- Updated CHANGELOG with v0.2.11 and v0.3.0 entries
- Pinned `transformers<5.0` to avoid breaking changes

## [0.2.11] - 2026-03-08

### Changes
- Merged Job Queue into Experiments page
- Fixed adapter polling for completed runs
- Removed unused variable to pass ruff lint

## [0.1.0] - 2026-03-05

Initial open-source release.

### Training
- LoRA and QLoRA (4-bit) fine-tuning on Apple Silicon via MLX
- DPO (Direct Preference Optimization) training
- Sequence packing for 2-5x throughput improvement
- Gradient checkpointing for 40-60% memory savings
- Compiled training loop with gradient accumulation
- Cosine, linear, step, and exponential LR schedules with warmup
- Checkpoint resume with stateless LR reconstruction

### Models
- Llama 2/3 (all sizes)
- Mistral (mapped to Llama architecture)
- Qwen 2/3/3.5
- Phi-3/4
- Gemma 1/2/3 (1B-27B)
- Automatic Hugging Face model downloading and caching

### Studio UI
- Browser-based training dashboard (React + FastAPI)
- Real-time loss curve visualization via WebSocket
- Model library and dataset browser
- Interactive playground for testing fine-tuned models

### Data
- 20+ curated datasets across 7 categories
- Auto-detection of chat, completions, text, and preference formats
- Multi-dataset mixing with weighted sampling
- Data validation CLI with train/val overlap detection
- Arrow-based storage backend for tokenized datasets

### CLI
- `mlx-forge train` — Run training from YAML config
- `mlx-forge generate` — Text generation with optional LoRA adapters
- `mlx-forge prepare` — Pre-tokenize datasets
- `mlx-forge studio` — Launch browser-based UI
- `mlx-forge data` — Dataset management (catalog, download, import, validate)
