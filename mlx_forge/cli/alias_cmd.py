"""CLI command: alias — manage model aliases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ALIASES_PATH = Path("~/.mlxforge/aliases.json").expanduser()


def _load_aliases() -> dict[str, str]:
    """Load aliases from disk."""
    if ALIASES_PATH.exists():
        with open(ALIASES_PATH) as f:
            return json.load(f)
    return {}


def _save_aliases(aliases: dict[str, str]) -> None:
    """Save aliases to disk."""
    ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALIASES_PATH, "w") as f:
        json.dump(aliases, f, indent=2)


def run_alias(args) -> None:
    """Execute the alias command."""
    if args.alias_command == "set":
        aliases = _load_aliases()
        aliases[args.name] = args.model_id
        _save_aliases(aliases)
        print(f"Alias '{args.name}' → '{args.model_id}'")

    elif args.alias_command == "list":
        aliases = _load_aliases()
        if not aliases:
            print("No aliases configured.")
            print("  Set one with: mlx-forge alias set <name> <model-id>")
            return
        for name, model_id in sorted(aliases.items()):
            print(f"  {name} → {model_id}")

    elif args.alias_command == "remove":
        aliases = _load_aliases()
        if args.name in aliases:
            del aliases[args.name]
            _save_aliases(aliases)
            print(f"Removed alias '{args.name}'")
        else:
            print(f"Alias '{args.name}' not found.")
            sys.exit(1)

    else:
        print("Usage: mlx-forge alias {set,list,remove}")
        sys.exit(1)
