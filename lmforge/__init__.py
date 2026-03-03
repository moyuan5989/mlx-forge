"""LMForge — LoRA SFT training framework for MLX on Apple Silicon."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from lmforge._version import __version__
from lmforge.data.formats import detect_format, validate_samples
from lmforge.data.preprocessing import tokenize_dataset
from lmforge.inference.engine import GenerationResult


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
        output: Ignored (kept for CLI compat). Storage is now in ~/.lmforge/datasets/
        name: Dataset name for the registry. If omitted, derived from filename.
        trust_remote_code: Trust remote code when loading tokenizer
        max_seq_length: Maximum sequence length
        mask_prompt: Mask prompt tokens from loss
        revision: Optional HF revision/commit hash

    Returns:
        Dict of statistics (sample count, total tokens, etc.)
    """
    from lmforge.data import backend
    from lmforge.models.resolve import resolve_model

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
    print(f"Saving to datasets backend...")
    path = backend.save_tokenized(dataset_name, model, tokenized)

    meta_path = path / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"  Preprocessed {meta['num_samples']} samples")
    print(f"  Total tokens: {meta['total_tokens']}")
    print(f"  Min/mean/max length: {meta['min_length']}/{meta['mean_length']:.1f}/{meta['max_length']}")

    return meta


def train(config, resume: str | None = None) -> "lmforge.trainer.state.TrainState":
    """Run LoRA SFT training from a config file or TrainingConfig object.

    Args:
        config: Path to a YAML config file (str) or a TrainingConfig instance.
        resume: Path to checkpoint directory to resume from.

    Returns:
        Final TrainState after training completes.
    """
    import mlx.core as mx
    import yaml

    from lmforge.adapters.lora import apply_lora
    from lmforge.adapters.targeting import get_patterns, resolve_targets
    from lmforge.config import TrainingConfig
    from lmforge.data import backend
    from lmforge.manifest import write_manifest
    from lmforge.models.loader import load_model
    from lmforge.models.resolve import resolve_model
    from lmforge.trainer.callbacks import ConsoleCallback, MetricsLoggerCallback
    from lmforge.trainer.trainer import Trainer

    # Load config if it's a path
    if isinstance(config, str):
        config = TrainingConfig.from_yaml(config)

    print(f"LMForge v0 — Training")
    print(f"Model: {config.model.path}")
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
    from lmforge.trainer.checkpoint import CheckpointManager
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

    # Quantize model if configured (QLoRA: quantize THEN apply LoRA)
    if config.model.quantization:
        from lmforge.models.quantize import quantize_model
        quantize_model(model, config.model.quantization)
        print(f"Quantized to {config.model.quantization.bits}-bit "
              f"(group_size={config.model.quantization.group_size})")
        print()

    # Apply LoRA adapters
    print("Applying LoRA adapters...")
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

    train_name, train_dataset = _load_or_prepare(config.data.train, "training")
    _, val_dataset = _load_or_prepare(config.data.valid, "validation")
    print()

    # Compute fingerprint for manifest (backward compat)
    train_fingerprint = backend.compute_fingerprint(config.data.train, tokenizer)

    # Write manifest
    print("Writing manifest...")
    manifest = write_manifest(
        run_dir,
        config.model_dump(),
        train_fingerprint,
        resolved_model.resolution_metadata,
    )
    print(f"Manifest written: {run_dir / 'manifest.json'}")
    print()

    # Create callbacks
    callbacks = [
        ConsoleCallback(num_iters=config.training.num_iters),
        MetricsLoggerCallback(log_path=run_dir / "logs" / "metrics.jsonl"),
    ]

    # Add WandB callback if configured
    if hasattr(config.training, 'wandb_project') and config.training.wandb_project:
        try:
            from lmforge.trainer.callbacks import WandBCallback
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

    # Create trainer (SFT or DPO based on training_type)
    if config.training.training_type == "dpo":
        from lmforge.trainer.dpo_trainer import DPOTrainer
        trainer = DPOTrainer(
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

    required = ["adapters.safetensors", "optimizer.safetensors", "state.json"]
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
            f"supported version 1. Please upgrade LMForge."
        )
    if state["step"] >= config.training.num_iters:
        raise ValueError(
            f"Checkpoint is at step {state['step']} but training is configured "
            f"for {config.training.num_iters} iterations. "
            f"Increase 'num_iters' in your config to continue training."
        )


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
    from lmforge.inference.engine import (
        generate as _generate,
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
            if isinstance(prompt_tokens, dict):
                prompt_tokens = prompt_tokens["input_ids"]
        elif prompt is not None:
            prompt_tokens = tokenizer.encode(prompt)
        else:
            raise ValueError("Must provide either 'prompt' or 'messages'")

        def _stream():
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
                yield tokenizer.decode([token_id])

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
