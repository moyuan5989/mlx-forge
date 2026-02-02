"""LMForge — LoRA SFT training framework for MLX on Apple Silicon."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from lmforge._version import __version__
from lmforge.data.cache import check_cache, compute_fingerprint, read_cache, write_cache
from lmforge.data.formats import detect_format, validate_samples
from lmforge.data.preprocessing import tokenize_dataset


def prepare(
    data_path: str,
    model: str,
    output: str | None = None,
    *,
    trust_remote_code: bool = False,
    max_seq_length: int = 2048,
    mask_prompt: bool = True,
) -> dict:
    """Pre-tokenize a dataset and write a safetensors cache to disk.

    Args:
        data_path: Path to JSONL data file
        model: HuggingFace model ID or local path (for tokenizer)
        output: Output cache directory (default: ~/.lmforge/cache/preprocessed)
        trust_remote_code: Trust remote code when loading tokenizer
        max_seq_length: Maximum sequence length
        mask_prompt: Mask prompt tokens from loss

    Returns:
        Dict of statistics (sample count, total tokens, etc.)
    """
    # Default output directory
    if output is None:
        output = "~/.lmforge/cache/preprocessed"

    # Load tokenizer
    print(f"Loading tokenizer from {model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model,
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
        error_msg = "\n".join(errors[:10])  # Show first 10 errors
        if len(errors) > 10:
            error_msg += f"\n... and {len(errors) - 10} more errors"
        raise ValueError(f"Validation failed:\n{error_msg}")

    # Compute fingerprint
    fingerprint = compute_fingerprint(data_path, tokenizer)
    print(f"Data fingerprint: {fingerprint}")

    # Check cache
    if check_cache(output, fingerprint):
        print(f"Cache hit! Loading from {output}/{fingerprint}")
        meta_path = Path(output).expanduser() / fingerprint / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"✓ Loaded {meta['num_samples']} samples, {meta['total_tokens']} tokens")
        return meta

    # Cache miss - tokenize
    print(f"Cache miss. Tokenizing {len(samples)} samples...")
    tokenized = tokenize_dataset(
        samples,
        tokenizer,
        fmt,
        mask_prompt=mask_prompt,
        max_seq_length=max_seq_length,
    )

    # Write cache
    print(f"Writing cache to {output}/{fingerprint}...")
    meta = write_cache(output, fingerprint, tokenized, fmt)

    print(f"✓ Preprocessed {meta['num_samples']} samples")
    print(f"  Total tokens: {meta['total_tokens']}")
    print(f"  Min/mean/max length: {meta['min_length']}/{meta['mean_length']:.1f}/{meta['max_length']}")
    print(f"  Shards: {meta['num_shards']}")

    return meta


def train(config) -> "lmforge.trainer.state.TrainState":
    """Run LoRA SFT training from a config file or TrainingConfig object.

    Args:
        config: Path to a YAML config file (str) or a TrainingConfig instance.

    Returns:
        Final TrainState after training completes.
    """
    import json
    from pathlib import Path

    import mlx.core as mx
    import yaml

    from lmforge.adapters.lora import apply_lora
    from lmforge.adapters.targeting import get_patterns, resolve_targets
    from lmforge.config import TrainingConfig
    from lmforge.data.cache import check_cache, read_cache
    from lmforge.manifest import write_manifest
    from lmforge.models.loader import load_model
    from lmforge.trainer.callbacks import ConsoleCallback, MetricsLoggerCallback
    from lmforge.trainer.trainer import Trainer

    # Load config if it's a path
    if isinstance(config, str):
        config = TrainingConfig.from_yaml(config)

    print(f"LMForge v0 — Training")
    print(f"Model: {config.model.path}")
    print(f"Adapter: {config.adapter.method} (rank={config.adapter.rank})")
    print()

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
    model, tokenizer = load_model(config.model.path, config.model.tokenizer_path, config.model.trust_remote_code)
    print(f"Model loaded: {type(model).__name__}")
    print()

    # Apply LoRA adapters
    print("Applying LoRA adapters...")
    patterns = get_patterns(config.adapter)
    targets = resolve_targets(model, patterns, config.adapter.num_layers)
    print(f"Matched {len(targets)} modules")

    apply_lora(model, targets, config.adapter)
    trainable_params = sum(p.size for p in model.trainable_parameters().values())
    total_params = sum(p.size for p in model.parameters().values())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print()

    # Load or prepare training data
    print("Loading training data...")
    train_cache_meta = check_cache(config.data.train, config.model.path, config.data.cache_dir)
    if train_cache_meta is None:
        print(f"Cache miss for {config.data.train}. Running prepare...")
        train_cache_meta = prepare(config.data.train, config.model.path, config.data.cache_dir)
    else:
        print(f"Cache hit: {train_cache_meta['num_samples']} samples, {train_cache_meta['total_tokens']} tokens")

    train_dataset = read_cache(Path(config.data.cache_dir).expanduser() / train_cache_meta["fingerprint"])

    # Load validation data
    print("Loading validation data...")
    val_cache_meta = check_cache(config.data.valid, config.model.path, config.data.cache_dir)
    if val_cache_meta is None:
        print(f"Cache miss for {config.data.valid}. Running prepare...")
        val_cache_meta = prepare(config.data.valid, config.model.path, config.data.cache_dir)
    else:
        print(f"Cache hit: {val_cache_meta['num_samples']} samples, {val_cache_meta['total_tokens']} tokens")

    val_dataset = read_cache(Path(config.data.cache_dir).expanduser() / val_cache_meta["fingerprint"])
    print()

    # Write manifest
    print("Writing manifest...")
    manifest = write_manifest(run_dir, config.model_dump(), train_cache_meta["fingerprint"])
    print(f"Manifest written: {run_dir / 'manifest.json'}")
    print()

    # Create callbacks
    callbacks = [
        ConsoleCallback(num_iters=config.training.num_iters),
        MetricsLoggerCallback(log_path=run_dir / "logs" / "metrics.jsonl"),
    ]

    # Add WandB callback if configured
    if config.training.wandb_project:
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

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        callbacks=callbacks,
    )

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
