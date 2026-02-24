"""CLI handler for `lmforge studio` command."""

from __future__ import annotations


def run_studio(args):
    """Start the LMForge Studio server."""
    try:
        from lmforge.studio.server import run_server
    except ImportError as e:
        print(
            "LMForge Studio requires additional dependencies.\n"
            "Install them with: pip install lmforge[studio]\n"
            f"\nMissing: {e}"
        )
        raise SystemExit(1)

    run_server(
        host=args.host,
        port=args.port,
    )
