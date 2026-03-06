"""CLI entry point for LMForge.

Thin wrapper that parses args and delegates to library functions.
No business logic in CLI code.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmforge",
        description="LMForge — LoRA SFT training framework for MLX on Apple Silicon",
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
        help="Ignored (kept for compat). Storage is in ~/.lmforge/datasets/",
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
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer",
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

    # --- studio ---
    studio_parser = subparsers.add_parser(
        "studio",
        help="Start the LMForge Studio web UI server",
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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "prepare":
        from lmforge.cli.prepare_cmd import run_prepare

        run_prepare(args)
    elif args.command == "train":
        from lmforge.cli.train_cmd import run_train

        run_train(args)
    elif args.command == "generate":
        from lmforge.cli.generate_cmd import run_generate

        run_generate(args)
    elif args.command == "data":
        from lmforge.cli.data_cmd import run_data

        run_data(args)
    elif args.command == "studio":
        from lmforge.cli.studio_cmd import run_studio

        run_studio(args)
    else:
        parser.print_help()
        sys.exit(1)


def _get_version() -> str:
    from lmforge._version import __version__

    return __version__


if __name__ == "__main__":
    main()
