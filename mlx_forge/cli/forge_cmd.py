"""CLI command: forge — manage model bundles (base + adapter + config)."""

from __future__ import annotations

import sys


def run_forge(args) -> None:
    """Execute the forge command."""
    if args.forge_command == "create":
        _forge_create(args)
    elif args.forge_command == "list":
        _forge_list()
    elif args.forge_command == "delete":
        _forge_delete(args)
    elif args.forge_command == "show":
        _forge_show(args)
    else:
        print("Usage: mlx-forge forge {create,list,delete,show}")
        sys.exit(1)


def _forge_create(args) -> None:
    """Create a new forge."""
    from mlx_forge.forge import ForgeSpec

    if getattr(args, "from_run", None):
        try:
            forge = ForgeSpec.from_run(args.from_run)
            forge.name = args.name
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif getattr(args, "file", None):
        try:
            forge = ForgeSpec.from_yaml(args.file)
            forge.name = args.name
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif getattr(args, "base", None):
        forge = ForgeSpec(
            name=args.name,
            base=args.base,
            adapter=getattr(args, "adapter", None),
            system=getattr(args, "system", None),
        )
    else:
        print("Error: provide --from-run, --file, or --base")
        sys.exit(1)

    path = forge.save()
    print(f"Created forge '{forge.name}' at {path}")
    print(f"  Base: {forge.base}")
    if forge.adapter:
        print(f"  Adapter: {forge.adapter}")
    if forge.system:
        print(f"  System: {forge.system[:60]}...")
    print()
    print(f"Serve with: mlx-forge serve --model forge:{forge.name}")


def _forge_list() -> None:
    """List all forges."""
    from mlx_forge.forge import list_forges

    forges = list_forges()
    if not forges:
        print("No forges found.")
        print("  Create one with: mlx-forge forge create <name> --from-run <run-id>")
        return

    for forge in forges:
        adapter_info = f" + {forge.adapter}" if forge.adapter else ""
        print(f"  {forge.name}: {forge.base}{adapter_info}")


def _forge_delete(args) -> None:
    """Delete a forge."""
    from mlx_forge.forge import delete_forge

    if delete_forge(args.name):
        print(f"Deleted forge '{args.name}'")
    else:
        print(f"Forge '{args.name}' not found.")
        sys.exit(1)


def _forge_show(args) -> None:
    """Show forge details."""
    from mlx_forge.forge import get_forge

    forge = get_forge(args.name)
    if not forge:
        print(f"Forge '{args.name}' not found.")
        sys.exit(1)

    print(f"Name:    {forge.name}")
    print(f"Base:    {forge.base}")
    print(f"Adapter: {forge.adapter or '(none)'}")
    if forge.system:
        print(f"System:  {forge.system[:100]}")
    if forge.parameters:
        print(f"Params:  {forge.parameters}")
