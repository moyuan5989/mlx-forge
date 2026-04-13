"""CLI entry point for MLX Forge.

Thin wrapper that parses args and delegates to library functions.
No business logic in CLI code.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-forge",
        description="MLX Forge — Fine-tune, experiment with, and run LLMs locally on your Mac",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- prepare ---
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Pre-tokenize a dataset and save for training",
    )
    prepare_parser.add_argument(
        "--data", required=True, help="Path to JSONL data file"
    )
    prepare_parser.add_argument(
        "--model", required=True, help="HuggingFace model ID or local path"
    )
    prepare_parser.add_argument(
        "--output",
        default=None,
        help="Ignored (kept for compat). Storage is in ~/.mlxforge/datasets/",
    )
    prepare_parser.add_argument(
        "--name",
        default=None,
        help="Dataset name for the registry (default: derived from filename)",
    )
    prepare_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    prepare_parser.add_argument(
        "--no-mask-prompt",
        action="store_true",
        help="Do not mask prompt tokens from loss",
    )
    prepare_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer",
    )

    # --- train ---
    train_parser = subparsers.add_parser(
        "train",
        help="Run LoRA SFT training from a config file",
    )
    train_parser.add_argument(
        "--config", required=True, help="Path to training YAML config file"
    )
    train_parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint directory to resume from",
    )

    # --- generate ---
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate text with optional LoRA adapter",
    )
    gen_parser.add_argument(
        "--model", required=True, help="HuggingFace model ID or local path"
    )
    gen_parser.add_argument(
        "--adapter",
        default=None,
        help="Path to checkpoint directory with adapters.safetensors",
    )
    gen_parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt (omit for interactive chat mode)",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 = greedy, default: 0.7)",
    )
    gen_parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold (default: 0.9)",
    )
    gen_parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    gen_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (1.0 = disabled, default: 1.0)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible generation",
    )
    gen_parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 = disabled, default: 0)",
    )
    gen_parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Min-p sampling threshold (0.0 = disabled, default: 0.0)",
    )
    gen_parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Frequency penalty (0.0 = disabled, default: 0.0)",
    )
    gen_parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty (0.0 = disabled, default: 0.0)",
    )
    gen_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer",
    )
    gen_parser.add_argument(
        "--draft-model",
        default=None,
        help="Draft model for speculative decoding",
    )
    gen_parser.add_argument(
        "--num-draft",
        type=int,
        default=5,
        help="Draft tokens per speculative decoding step (default: 5)",
    )
    gen_parser.add_argument(
        "--prompt-cache",
        default=None,
        help="Path to prompt cache file (save/load KV cache state)",
    )
    gen_parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable vision model support (requires mlx-vlm)",
    )

    # --- data ---
    data_parser = subparsers.add_parser(
        "data",
        help="Dataset management: catalog, download, import, inspect",
    )
    data_subs = data_parser.add_subparsers(dest="data_command", help="Data subcommands")

    # data list
    data_subs.add_parser("list", help="List downloaded datasets")

    # data catalog
    cat_parser = data_subs.add_parser("catalog", help="Show curated dataset catalog")
    cat_parser.add_argument(
        "--category",
        default=None,
        help="Filter by category (general, code, math, conversation, reasoning, safety, domain)",
    )

    # data download
    dl_parser = data_subs.add_parser("download", help="Download a dataset from the catalog")
    dl_parser.add_argument("dataset_id", help="Dataset ID from the catalog")
    dl_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (useful for Apple Silicon)",
    )

    # data import
    imp_parser = data_subs.add_parser("import", help="Import a local JSONL file")
    imp_parser.add_argument("file", help="Path to JSONL file")
    imp_parser.add_argument("--name", required=True, help="Name for the dataset")
    imp_parser.add_argument(
        "--format",
        default=None,
        choices=["chat", "completions", "text", "preference"],
        help="Override auto-detected format",
    )

    # data inspect
    insp_parser = data_subs.add_parser("inspect", help="Preview samples from a dataset")
    insp_parser.add_argument("name", help="Dataset name")
    insp_parser.add_argument(
        "--n", type=int, default=5, help="Number of samples to show (default: 5)"
    )

    # data stats
    stats_parser = data_subs.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("name", help="Dataset name")

    # data validate
    val_parser = data_subs.add_parser("validate", help="Validate a JSONL data file")
    val_parser.add_argument("file", help="Path to JSONL file to validate")
    val_parser.add_argument(
        "--val",
        default=None,
        help="Optional validation file for overlap detection",
    )

    # data delete
    del_parser = data_subs.add_parser("delete", help="Delete a downloaded dataset")
    del_parser.add_argument("name", help="Dataset name")

    # data hf-import
    hf_parser = data_subs.add_parser("hf-import", help="Import a HuggingFace dataset")
    hf_parser.add_argument("dataset_id", help="HuggingFace dataset ID (e.g., 'tatsu-lab/alpaca')")
    hf_parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    hf_parser.add_argument("--subset", default=None, help="Dataset subset/config name")
    hf_parser.add_argument("--name", default=None, help="Local name for the dataset")
    hf_parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")

    # --- export ---
    export_parser = subparsers.add_parser(
        "export",
        help="Merge LoRA adapters into base model and save",
    )
    export_parser.add_argument(
        "--run-id", required=True, help="Run ID to export (from ~/.mlxforge/runs/)"
    )
    export_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ~/.mlxforge/exports/{run-id})",
    )
    export_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint name (default: best or latest)",
    )
    export_parser.add_argument(
        "--format",
        choices=["safetensors", "gguf"],
        default="safetensors",
        help="Export format (default: safetensors)",
    )
    export_parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Push to HuggingFace Hub (e.g., 'username/model-name')",
    )
    export_parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="Upload only adapter files (not full merged model)",
    )
    export_parser.add_argument(
        "--private",
        action="store_true",
        help="Create private HuggingFace repository",
    )
    export_parser.add_argument(
        "--quantize",
        choices=["f16", "f32", "q8_0", "q4_0"],
        default=None,
        help="Quantize GGUF output (requires --format gguf)",
    )

    # --- encode ---
    encode_parser = subparsers.add_parser(
        "encode",
        help="Extract embeddings from an encoder model (BERT, RoBERTa, DeBERTa)",
    )
    encode_parser.add_argument(
        "--model", required=True, help="HuggingFace model ID or local path"
    )
    encode_parser.add_argument(
        "--texts",
        nargs="+",
        required=True,
        help="Text strings to encode",
    )
    encode_parser.add_argument(
        "--adapter",
        default=None,
        help="Path to adapter checkpoint directory",
    )
    encode_parser.add_argument(
        "--pooling",
        choices=["cls", "mean"],
        default="cls",
        help="Pooling strategy (default: cls)",
    )
    encode_parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of embeddings",
    )
    encode_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer",
    )

    # --- studio ---
    studio_parser = subparsers.add_parser(
        "studio",
        help="Start the MLX Forge Studio web UI server",
    )
    studio_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    studio_parser.add_argument(
        "--port",
        type=int,
        default=8741,
        help="Port number (default: 8741)",
    )

    # --- serve ---
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start an OpenAI-compatible API server for local models",
    )
    serve_parser.add_argument(
        "--model",
        default=None,
        help="Model to pre-load (run ID, export name, HF repo, or local path)",
    )
    serve_parser.add_argument(
        "--adapter",
        default=None,
        help="Path to adapter checkpoint directory",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number (default: 8000)",
    )
    serve_parser.add_argument(
        "--draft-model",
        default=None,
        help="Draft model for speculative decoding",
    )
    serve_parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable vision model support (requires mlx-vlm)",
    )
    serve_parser.add_argument(
        "--max-models",
        type=int,
        default=1,
        help="Maximum models to keep in memory (default: 1)",
    )
    serve_parser.add_argument(
        "--keep-alive",
        default="5m",
        help="Default model keep-alive duration (default: 5m). "
        "Supports: '5m', '1h', '30s', 300, -1 (forever), 0 (immediate unload)",
    )
    serve_parser.add_argument(
        "--aliases",
        default=None,
        help="Path to aliases JSON file (default: ~/.mlxforge/aliases.json)",
    )
    serve_parser.add_argument(
        "--context-length",
        type=int,
        default=0,
        help="Default context window size (0 = auto, default: 0). "
        "When set, uses rotating cache to handle overflow.",
    )
    serve_parser.add_argument(
        "--num-keep",
        type=int,
        default=0,
        help="Tokens to always keep at start of context on overflow (default: 0). "
        "Use to preserve system prompt tokens.",
    )

    # --- alias ---
    alias_parser = subparsers.add_parser(
        "alias",
        help="Manage model aliases",
    )
    alias_subs = alias_parser.add_subparsers(dest="alias_command", help="Alias subcommands")

    alias_set = alias_subs.add_parser("set", help="Set an alias")
    alias_set.add_argument("name", help="Alias name")
    alias_set.add_argument("model_id", help="Model ID the alias points to")

    alias_subs.add_parser("list", help="List all aliases")

    alias_rm = alias_subs.add_parser("remove", help="Remove an alias")
    alias_rm.add_argument("name", help="Alias name to remove")

    # --- train (add --vision flag) ---
    train_parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable vision model training (requires mlx-vlm)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "prepare":
        from mlx_forge.cli.prepare_cmd import run_prepare

        run_prepare(args)
    elif args.command == "train":
        from mlx_forge.cli.train_cmd import run_train

        run_train(args)
    elif args.command == "generate":
        from mlx_forge.cli.generate_cmd import run_generate

        run_generate(args)
    elif args.command == "data":
        from mlx_forge.cli.data_cmd import run_data

        run_data(args)
    elif args.command == "export":
        from mlx_forge.cli.export_cmd import run_export

        run_export(args)
    elif args.command == "encode":
        from mlx_forge.inference.encoder import encode
        from mlx_forge.inference.engine import load_for_inference

        model, tokenizer = load_for_inference(
            args.model,
            adapter_path=args.adapter,
            trust_remote_code=args.trust_remote_code,
        )
        embeddings = encode(
            model,
            tokenizer,
            args.texts,
            pooling=args.pooling,
            normalize=not args.no_normalize,
        )
        for i, emb in enumerate(embeddings):
            print(f"[{i}] dim={emb.shape[0]}, norm={float((emb * emb).sum() ** 0.5):.4f}")
            print(f"    {emb.tolist()[:5]}...")
    elif args.command == "studio":
        from mlx_forge.cli.studio_cmd import run_studio

        run_studio(args)
    elif args.command == "serve":
        from mlx_forge.cli.serve_cmd import run_serve

        run_serve(args)
    elif args.command == "alias":
        from mlx_forge.cli.alias_cmd import run_alias

        run_alias(args)
    else:
        parser.print_help()
        sys.exit(1)


def _get_version() -> str:
    from mlx_forge._version import __version__

    return __version__


if __name__ == "__main__":
    main()
