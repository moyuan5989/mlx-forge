"""CLI command: export — merge LoRA adapters into base model and save."""

from __future__ import annotations

import sys
from pathlib import Path


def run_export(args) -> None:
    """Load base model, apply adapter, fuse, and save merged model."""
    run_id = args.run_id
    output_dir = args.output_dir
    checkpoint = args.checkpoint

    # Find run directory
    runs_dir = Path("~/.mlxforge/runs").expanduser()
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        print(f"Error: run '{run_id}' not found at {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Load config
    import yaml

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = config.get("model", {}).get("path")
    if not model_path:
        print("Error: no model path found in config", file=sys.stderr)
        sys.exit(1)

    # Find checkpoint
    ckpt_dir = run_dir / "checkpoints"
    if checkpoint:
        ckpt_path = ckpt_dir / checkpoint
    else:
        # Try "best" symlink first, then latest
        best_link = ckpt_dir / "best"
        if best_link.exists():
            ckpt_path = best_link.resolve()
        else:
            ckpts = sorted(
                [d for d in ckpt_dir.iterdir() if d.is_dir() and not d.is_symlink()],
            )
            if not ckpts:
                print(f"Error: no checkpoints found in {ckpt_dir}", file=sys.stderr)
                sys.exit(1)
            ckpt_path = ckpts[-1]

    # Check adapter method
    adapter_config = config.get("adapter", {})
    adapter_method = adapter_config.get("method", "lora")

    # Default output directory
    if output_dir is None:
        output_dir = str(Path("~/.mlxforge/exports").expanduser() / run_id)

    print(f"Exporting run: {run_id}")
    print(f"Model: {model_path}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Output: {output_dir}")

    # Load model
    from mlx_forge.models.loader import load_model

    model, tokenizer_path = load_model(
        model_path,
        trust_remote_code=config.get("model", {}).get("trust_remote_code", False),
    )

    if adapter_method == "full":
        # Full fine-tuning: load model weights directly (no LoRA fuse needed)
        model_file = ckpt_path / "model.safetensors"
        if not model_file.exists():
            print(f"Error: model.safetensors not found at {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        from safetensors.mlx import load_file

        model_weights = load_file(str(model_file))
        model.load_weights(list(model_weights.items()))

        # Save directly (no fuse step needed)
        from mlx_forge.adapters.fuse import save_fused_model

        save_fused_model(model, tokenizer_path, output_dir)
    else:
        # LoRA/DoRA: load adapters, apply, fuse
        adapter_file = ckpt_path / "adapters.safetensors"
        if not adapter_file.exists():
            print(f"Error: adapters.safetensors not found at {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        from safetensors.mlx import load_file

        adapter_weights = load_file(str(adapter_file))

        # Apply LoRA targets from config
        from mlx_forge.adapters.targeting import resolve_targets
        from mlx_forge.config import AdapterConfig

        adapter_cfg = AdapterConfig(
            rank=adapter_config.get("rank", 8),
            scale=adapter_config.get("scale", 20.0),
            dropout=adapter_config.get("dropout", 0.0),
            preset=adapter_config.get("preset"),
            targets=adapter_config.get("targets"),
        )

        targets = resolve_targets(model, adapter_cfg)

        from mlx_forge.adapters.lora import apply_lora

        apply_lora(model, targets, adapter_cfg)

        # Load adapter weights into model
        model.load_weights(list(adapter_weights.items()))

        # Fuse
        from mlx_forge.adapters.fuse import fuse_model, save_fused_model

        model = fuse_model(model)

        # Save
        save_fused_model(model, tokenizer_path, output_dir)

    print("Export complete!")

    # GGUF conversion if requested
    if getattr(args, "format", "safetensors") == "gguf":
        from mlx_forge.export.gguf_writer import convert_to_gguf
        gguf_output = Path(output_dir) / "model.gguf"
        print(f"\nConverting to GGUF: {gguf_output}")
        convert_to_gguf(output_dir, gguf_output)

    # Push to HuggingFace Hub if requested
    if getattr(args, "push_to_hub", None):
        from mlx_forge.hub.upload import generate_model_card, push_to_hub

        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")

        # Generate model card
        model_card = generate_model_card(config, base_model=model_path)
        (Path(output_dir) / "README.md").write_text(model_card)

        url = push_to_hub(
            output_dir,
            args.push_to_hub,
            adapter_only=getattr(args, "adapter_only", False),
            private=getattr(args, "private", False),
        )
        print(f"Uploaded to: {url}")
