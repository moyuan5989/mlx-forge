"""MLX Forge — LoRA SFT training framework for MLX on Apple Silicon."""

import json
import os
from pathlib import Path

# Suppress "PyTorch was not found..." warning emitted during transformers import.
# transformers overrides logger config internally, so the env var is the only
# reliable way to control verbosity before the library initializes.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer  # noqa: E402

from mlx_forge._version import __version__ as __version__  # noqa: E402
from mlx_forge.data.formats import detect_format, validate_samples  # noqa: E402
from mlx_forge.data.preprocessing import tokenize_dataset  # noqa: E402
from mlx_forge.inference.engine import GenerationResult  # noqa: E402


def prepare(
    data_path: str,
    model: str,
    output: str | None = None,
    *,
    name: str | None = None,
    trust_remote_code: bool = False,
    max_seq_length: int = 2048,
    mask_prompt: bool = True,
    revision: str | None = None,
) -> dict:
    """Pre-tokenize a dataset and save as Arrow dataset for memory-mapped access.

    Args:
        data_path: Path to JSONL data file
        model: HuggingFace model ID or local path (for tokenizer)
        output: Ignored (kept for CLI compat). Storage is now in ~/.mlxforge/datasets/
        name: Dataset name for the registry. If omitted, derived from filename.
        trust_remote_code: Trust remote code when loading tokenizer
        max_seq_length: Maximum sequence length
        mask_prompt: Mask prompt tokens from loss
        revision: Optional HF revision/commit hash

    Returns:
        Dict of statistics (sample count, total tokens, etc.)
    """
    from mlx_forge.data import backend
    from mlx_forge.models.resolve import resolve_model

    # Resolve model (HF repo ID -> local path)
    print(f"Resolving model: {model}...")
    resolved = resolve_model(
        model,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    print()

    # Load tokenizer
    print(f"Loading tokenizer from {resolved.local_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved.local_path,
        trust_remote_code=trust_remote_code,
    )

    # Read JSONL
    print(f"Reading {data_path}...")
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path_obj) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if not samples:
        raise ValueError(f"No samples found in {data_path}")

    # Detect format
    fmt = detect_format(samples)
    print(f"Detected format: {fmt}")

    # Validate samples
    print(f"Validating {len(samples)} samples...")
    errors = validate_samples(samples, fmt)
    if errors:
        error_msg = "\n".join(errors[:10])
        if len(errors) > 10:
            error_msg += f"\n... and {len(errors) - 10} more errors"
        raise ValueError(f"Validation failed:\n{error_msg}")

    # Derive dataset name from filename if not provided
    dataset_name = name or data_path_obj.stem

    # Check if already processed
    if backend.tokenized_exists(dataset_name, model):
        print(f"Already processed: {dataset_name} for {model}")
        path = backend.get_processed_path(dataset_name, model)
        meta_path = path / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  {meta['num_samples']} samples, {meta['total_tokens']} tokens")
        return meta

    # Tokenize
    print(f"Tokenizing {len(samples)} samples...")
    tokenized = tokenize_dataset(
        samples,
        tokenizer,
        fmt,
        mask_prompt=mask_prompt,
        max_seq_length=max_seq_length,
    )

    # Save via datasets backend
    print("Saving to datasets backend...")
    path = backend.save_tokenized(dataset_name, model, tokenized)

    meta_path = path / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"  Preprocessed {meta['num_samples']} samples")
    print(f"  Total tokens: {meta['total_tokens']}")
    print(f"  Min/mean/max length: {meta['min_length']}/{meta['mean_length']:.1f}/{meta['max_length']}")

    return meta


def train(config, resume: str | None = None):  # -> TrainState
    """Run LoRA SFT training from a config file or TrainingConfig object.

    Args:
        config: Path to a YAML config file (str) or a TrainingConfig instance.
        resume: Path to checkpoint directory to resume from.

    Returns:
        Final TrainState after training completes.
    """
    import yaml

    from mlx_forge.adapters.lora import apply_lora
    from mlx_forge.adapters.targeting import get_patterns, resolve_targets
    from mlx_forge.config import TrainingConfig
    from mlx_forge.data import backend
    from mlx_forge.manifest import write_manifest
    from mlx_forge.models.loader import load_model
    from mlx_forge.models.resolve import resolve_model
    from mlx_forge.trainer.callbacks import ConsoleCallback, MetricsLoggerCallback
    from mlx_forge.trainer.trainer import Trainer

    # Load config if it's a path
    if isinstance(config, str):
        config = TrainingConfig.from_yaml(config)

    import mlx.core as mx

    # Set wired memory limit early, before loading model weights.
    # Reserve 25% of recommended limit for OS + apps to prevent system freeze.
    # On unified memory Macs, wiring too much starves macOS of physical RAM.
    if mx.metal.is_available():
        device_info = mx.device_info()
        if "max_recommended_working_set_size" in device_info:
            safe_limit = int(device_info["max_recommended_working_set_size"] * 0.75)
            mx.set_wired_limit(safe_limit)

    print("MLX Forge v0 — Training")
    print(f"Model: {config.model.path}")
    if config.adapter.method == "full":
        print("Adapter: full (all parameters)")
    else:
        print(f"Adapter: {config.adapter.method} (rank={config.adapter.rank})")
    print()

    # Resolve model (HF repo ID -> local path)
    print("Resolving model...")
    resolved_model = resolve_model(
        config.model.path,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    print()

    # Resolve tokenizer if separate path specified
    if config.model.tokenizer_path:
        print("Resolving tokenizer...")
        resolved_tokenizer = resolve_model(
            config.model.tokenizer_path,
            trust_remote_code=config.model.trust_remote_code,
        )
        tokenizer_path = resolved_tokenizer.local_path
        print()
    else:
        tokenizer_path = None

    # Create run directory
    from mlx_forge.trainer.checkpoint import CheckpointManager
    manager = CheckpointManager(config)
    run_dir = manager.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print()

    # Write config.yaml
    (run_dir / "config.yaml").write_text(yaml.dump(config.model_dump(), default_flow_style=False))

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model(
        resolved_model.local_path,
        tokenizer_path=tokenizer_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    print(f"Model loaded: {type(model).__name__}")
    print()

    # Auto-enable gradient checkpointing for memory-hungry architectures
    # Qwen3.5 DeltaNet layers use float32 recurrence which 2-3x memory vs standard attention
    model_type = getattr(getattr(model, 'args', None), 'model_type', '')
    if model_type in ('qwen3_5',) and not config.training.gradient_checkpointing:
        print("Auto-enabling gradient checkpointing for Qwen3.5 (DeltaNet layers require float32)")
        config.training.gradient_checkpointing = True

    # Full fine-tuning: validate and skip LoRA
    if config.adapter.method == "full":
        if config.model.quantization:
            raise ValueError(
                "Full fine-tuning is incompatible with quantization. "
                "Remove 'quantization' from config or use method: 'lora'/'dora'."
            )
        print("Full fine-tuning mode — all parameters trainable")
        print("WARNING: Full FT uses 2-3x more memory than LoRA. "
              "Consider enabling gradient_checkpointing: true")
        print()
    else:
        # Quantize model if configured (QLoRA: quantize THEN apply LoRA)
        if config.model.quantization:
            from mlx_forge.models.quantize import quantize_model
            quantize_model(model, config.model.quantization)
            print(f"Quantized to {config.model.quantization.bits}-bit "
                  f"(group_size={config.model.quantization.group_size})")
            print()

        # Freeze base model before applying LoRA — only LoRA params should be trainable.
        # For QLoRA, quantize_model() already calls model.freeze(). For fp16 LoRA, we
        # need to freeze explicitly. apply_lora() then creates new unfrozen LoRA params.
        if not config.model.quantization:
            model.freeze()

        # Apply LoRA/DoRA adapters
        print(f"Applying {config.adapter.method.upper()} adapters...")
        patterns = get_patterns(config.adapter)
        targets = resolve_targets(model, patterns, config.adapter.num_layers)
        print(f"Matched {len(targets)} modules")

        apply_lora(model, targets, config.adapter)

    # Count parameters
    from mlx.utils import tree_flatten
    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print()

    # Enable gradient checkpointing if configured
    if config.training.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
        print("Gradient checkpointing enabled")
        print()

    # Pre-flight memory safety check
    _check_memory_safety(total_params, config)

    # Load or prepare training and validation data
    tokenizer_for_data = config.model.tokenizer_path or config.model.path

    def _load_or_prepare(data_path: str, label: str):
        """Load data from backend or run prepare if not cached."""
        print(f"Loading {label} data...")
        dataset_name = Path(data_path).stem

        if backend.tokenized_exists(dataset_name, config.model.path):
            print(f"Cache hit: {dataset_name}")
            ds = backend.load_tokenized(dataset_name, config.model.path)
            print(f"  {len(ds)} samples (memory-mapped)")
            return dataset_name, ds
        else:
            print(f"Cache miss for {data_path}. Running prepare...")
            prepare(
                data_path,
                tokenizer_for_data,
                name=dataset_name,
                trust_remote_code=config.model.trust_remote_code,
                max_seq_length=config.data.max_seq_length,
                mask_prompt=config.data.mask_prompt,
            )
            ds = backend.load_tokenized(dataset_name, config.model.path)
            return dataset_name, ds

    # Handle HuggingFace dataset if configured
    if config.data.hf_dataset:
        from mlx_forge.data.hf_loader import load_hf_dataset, save_as_jsonl

        print(f"Loading HuggingFace dataset: {config.data.hf_dataset}")
        samples = load_hf_dataset(
            config.data.hf_dataset,
            split=config.data.hf_split,
            subset=config.data.hf_subset,
            columns=config.data.hf_columns,
            max_samples=config.data.hf_max_samples,
        )
        print(f"Loaded {len(samples)} samples")

        # Save as JSONL
        raw_dir = Path("~/.mlxforge/datasets/raw").expanduser()
        dataset_name = config.data.hf_dataset.replace("/", "_")
        train_path = raw_dir / f"{dataset_name}_train.jsonl"
        save_as_jsonl(samples, train_path)

        # Auto-split if no validation set specified
        if config.data.valid is None:
            split_idx = int(len(samples) * 0.9)
            train_samples = samples[:split_idx]
            val_samples = samples[split_idx:]
            save_as_jsonl(train_samples, train_path)
            val_path = raw_dir / f"{dataset_name}_val.jsonl"
            save_as_jsonl(val_samples, val_path)
            config.data.train = str(train_path)
            config.data.valid = str(val_path)
        else:
            config.data.train = str(train_path)

        print(f"Saved to {train_path}")

    # Multi-source mixing or single dataset
    if config.data.sources:
        from mlx_forge.data.mixing import MixedDatasetIterator

        source_datasets = []
        source_weights = []
        for src in config.data.sources:
            data_path = src.path or src.dataset
            _, ds = _load_or_prepare(data_path, f"source ({data_path})")
            source_datasets.append(ds)
            source_weights.append(src.weight)

        train_dataset = MixedDatasetIterator(
            source_datasets, source_weights,
            seed=config.training.seed,
        )
        train_name = "mixed"
        train_fingerprint = "mixed"
        print(f"Mixed dataset: {len(config.data.sources)} sources")
    else:
        train_name, train_dataset = _load_or_prepare(config.data.train, "training")
        train_fingerprint = backend.compute_fingerprint(config.data.train, tokenizer)

    _, val_dataset = _load_or_prepare(config.data.valid, "validation")
    print()

    # Write manifest
    print("Writing manifest...")
    write_manifest(
        run_dir,
        config.model_dump(),
        train_fingerprint,
        resolved_model.resolution_metadata,
    )
    print(f"Manifest written: {run_dir / 'manifest.json'}")
    print()

    # Create callbacks
    from mlx_forge.trainer.callbacks import HeartbeatCallback
    callbacks = [
        ConsoleCallback(num_iters=config.training.num_iters),
        MetricsLoggerCallback(log_path=run_dir / "logs" / "metrics.jsonl"),
        HeartbeatCallback(run_dir=run_dir),
    ]

    # Add WandB callback if configured
    if hasattr(config.training, 'wandb_project') and config.training.wandb_project:
        try:
            from mlx_forge.trainer.callbacks import WandBCallback
            callbacks.append(
                WandBCallback(
                    project=config.training.wandb_project,
                    run_name=run_dir.name,
                    config=config.model_dump(),
                )
            )
            print("WandB logging enabled")
        except ImportError:
            print("Warning: wandb not installed, skipping WandB logging")

    # Handle streaming data if configured (M33)
    if config.data.streaming:
        from mlx_forge.data.streaming import StreamingHFDataset, StreamingJSONLDataset

        if config.data.hf_dataset:
            train_dataset = StreamingHFDataset(
                config.data.hf_dataset,
                split=config.data.hf_split,
                tokenizer=tokenizer,
                subset=config.data.hf_subset,
                columns=config.data.hf_columns,
                max_seq_length=config.data.max_seq_length,
                mask_prompt=config.data.mask_prompt,
            )
            print("Using streaming HF dataset")
        elif config.data.train:
            train_dataset = StreamingJSONLDataset(
                config.data.train,
                tokenizer=tokenizer,
                max_seq_length=config.data.max_seq_length,
                mask_prompt=config.data.mask_prompt,
            )
            print("Using streaming JSONL dataset")

    # Create trainer (SFT, DPO, GRPO, ORPO, KTO, or SimPO based on training_type)
    if config.training.training_type == "grpo":
        from mlx_forge.trainer.grpo_trainer import GRPOTrainer
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            callbacks=callbacks,
            checkpoint_manager=manager,
        )
    elif config.training.training_type == "dpo":
        from mlx_forge.trainer.dpo_trainer import DPOTrainer
        trainer = DPOTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            callbacks=callbacks,
            checkpoint_manager=manager,
        )
    elif config.training.training_type == "orpo":
        from mlx_forge.trainer.orpo_trainer import ORPOTrainer
        trainer = ORPOTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            callbacks=callbacks,
            checkpoint_manager=manager,
        )
    elif config.training.training_type == "kto":
        from mlx_forge.trainer.kto_trainer import KTOTrainer
        trainer = KTOTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            callbacks=callbacks,
            checkpoint_manager=manager,
        )
    elif config.training.training_type == "simpo":
        from mlx_forge.trainer.simpo_trainer import SimPOTrainer
        trainer = SimPOTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            callbacks=callbacks,
            checkpoint_manager=manager,
        )
    else:
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            callbacks=callbacks,
            checkpoint_manager=manager,
        )

    # Handle resume from checkpoint
    if resume:
        resume_path = Path(resume).expanduser()
        _validate_resume(resume_path, config)
        restored_state = manager.load(resume_path, model, trainer.optimizer)
        trainer.state = restored_state
        print(f"Resumed from {resume_path} at step {restored_state.step}")
        print()

    # Run training
    print("Starting training...")
    print()
    final_state = trainer.fit()

    print()
    print("Training complete!")
    print(f"Final step: {final_state.step}")
    print(f"Best validation loss: {final_state.best_val_loss:.4f}")
    print(f"Total tokens trained: {final_state.trained_tokens:,}")
    print(f"Checkpoints saved to: {run_dir / 'checkpoints'}")

    return final_state


def _validate_resume(resume_path: Path, config) -> None:
    """Validate that a checkpoint directory is compatible with the current config."""
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {resume_path}")

    # Full FT checkpoints use model.safetensors; LoRA uses adapters.safetensors
    has_model = (resume_path / "model.safetensors").exists()
    has_adapter = (resume_path / "adapters.safetensors").exists()
    if not has_model and not has_adapter:
        raise FileNotFoundError(
            f"Checkpoint missing model weights in {resume_path}. "
            f"Expected adapters.safetensors or model.safetensors."
        )
    required = ["optimizer.safetensors", "state.json"]
    missing = [f for f in required if not (resume_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint missing {', '.join(missing)} in {resume_path}. "
            f"Expected files: {', '.join(required)}"
        )

    state = json.loads((resume_path / "state.json").read_text())
    if state.get("schema_version", 1) > 1:
        raise ValueError(
            f"Checkpoint schema version {state['schema_version']} is newer than "
            f"supported version 1. Please upgrade MLX Forge."
        )
    if state["step"] >= config.training.num_iters:
        raise ValueError(
            f"Checkpoint is at step {state['step']} but training is configured "
            f"for {config.training.num_iters} iterations. "
            f"Increase 'num_iters' in your config to continue training."
        )


def _check_memory_safety(total_params: int, config) -> None:
    """Check estimated memory vs available RAM. Auto-adjust to prevent OOM.

    On Apple Silicon, running out of memory freezes the entire system
    (not just the process) because GPU and CPU share unified memory.
    We auto-reduce settings rather than just warning.
    """
    try:
        import math

        from mlx_forge.models.memory import HardwareProfile

        hw = HardwareProfile.detect()
        budget_gb = hw.training_budget_gb

        # Rough estimate: model weights (fp16) + activations + optimizer
        bytes_per_param = 2  # fp16
        if config.model.quantization:
            bytes_per_param = config.model.quantization.bits / 8
        weights_gb = (total_params * bytes_per_param) / (1024 ** 3)

        # Infer hidden_dim from param count (rough: params ≈ 12 * L * D^2)
        num_layers = 28  # conservative default
        hidden_dim = int((total_params / (12 * num_layers)) ** 0.5)

        def _estimate_activations(batch_size, seq_len, checkpointing):
            effective_layers = math.sqrt(num_layers) if checkpointing else num_layers
            return (batch_size * seq_len * hidden_dim * 2 * effective_layers * 8) / (1024 ** 3)

        batch_size = config.training.batch_size
        seq_len = config.data.max_seq_length
        checkpointing = config.training.gradient_checkpointing

        activations_gb = _estimate_activations(batch_size, seq_len, checkpointing)
        total_est = weights_gb + activations_gb + 0.5
        usage_pct = total_est / budget_gb if budget_gb > 0 else 1.0

        adjusted = False

        # Step 1: Auto-enable gradient checkpointing if over 80%
        if usage_pct > 0.80 and not checkpointing:
            config.training.gradient_checkpointing = True
            checkpointing = True
            activations_gb = _estimate_activations(batch_size, seq_len, True)
            total_est = weights_gb + activations_gb + 0.5
            usage_pct = total_est / budget_gb if budget_gb > 0 else 1.0
            print(f"Auto-enabled gradient checkpointing (estimated {total_est:.1f} GB / {budget_gb:.1f} GB budget)")
            adjusted = True

        # Step 2: Reduce batch_size if still over 90%
        while usage_pct > 0.90 and batch_size > 1:
            batch_size -= 1
            config.training.batch_size = batch_size
            # Compensate with gradient accumulation
            config.training.grad_accumulation_steps = max(
                config.training.grad_accumulation_steps,
                2,
            )
            activations_gb = _estimate_activations(batch_size, seq_len, checkpointing)
            total_est = weights_gb + activations_gb + 0.5
            usage_pct = total_est / budget_gb if budget_gb > 0 else 1.0
            print(f"Auto-reduced batch_size to {batch_size} "
                  f"(estimated {total_est:.1f} GB / {budget_gb:.1f} GB budget)")
            adjusted = True

        # Step 3: Hard warning if still over budget
        if usage_pct > 1.0:
            print(f"WARNING: Estimated memory ({total_est:.1f} GB) exceeds budget "
                  f"({budget_gb:.1f} GB). Training may freeze your Mac.")
            print("  Consider: reduce max_seq_length, enable QLoRA, or use a smaller model.")
            print()
        elif adjusted:
            print()
    except Exception:
        pass  # Non-critical — don't block training on estimation failure


def _enable_gradient_checkpointing(model) -> None:
    """Wrap each transformer layer's __call__ with mx.checkpoint."""
    import mlx.core as mx

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            layer.__call__ = mx.checkpoint(layer.__call__)


def generate(
    model: str,
    prompt: str | None = None,
    messages: list[dict] | None = None,
    *,
    adapter: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    repetition_penalty: float = 1.0,
    trust_remote_code: bool = False,
    seed: int | None = None,
    stream: bool = False,
) -> GenerationResult:
    """Generate text from a model with optional LoRA adapter."""
    from mlx_forge.inference.engine import (
        generate as _generate,
    )
    from mlx_forge.inference.engine import (
        generate_tokens,
        load_for_inference,
    )

    loaded_model, tokenizer = load_for_inference(
        model,
        adapter_path=adapter,
        trust_remote_code=trust_remote_code,
    )

    if stream:
        if messages is not None:
            prompt_tokens = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            if not isinstance(prompt_tokens, list):
                prompt_tokens = prompt_tokens["input_ids"]
        elif prompt is not None:
            prompt_tokens = tokenizer.encode(prompt)
        else:
            raise ValueError("Must provide either 'prompt' or 'messages'")

        def _stream():
            buffer = []
            for token_id in generate_tokens(
                loaded_model,
                prompt_tokens,
                tokenizer,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ):
                buffer.append(token_id)
                text = tokenizer.decode(buffer)
                if text and "\ufffd" not in text:
                    yield text
                    buffer.clear()
            if buffer:
                yield tokenizer.decode(buffer)

        return _stream()

    return _generate(
        loaded_model,
        tokenizer,
        prompt=prompt,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
