"""Handler for 'lmforge prepare' CLI command."""

from __future__ import annotations


def run_prepare(args) -> None:
    """Execute the prepare command from parsed CLI args."""
    from lmforge import prepare

    try:
        stats = prepare(
            data_path=args.data,
            model=args.model,
            output=args.output,
            name=getattr(args, 'name', None),
            trust_remote_code=args.trust_remote_code,
            max_seq_length=args.max_seq_length,
            mask_prompt=not args.no_mask_prompt,
        )
        print(f"\n  Prepared {stats['num_samples']} samples, {stats['total_tokens']} tokens")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise
