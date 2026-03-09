"""Handler for 'cortexlab data' CLI commands.

Subcommands:
  list       — List downloaded datasets
  catalog    — Show curated dataset catalog
  download   — Download a dataset from the catalog
  import     — Import a local JSONL file
  inspect    — Preview samples from a dataset
  stats      — Show dataset statistics
  delete     — Delete a downloaded dataset
"""

from __future__ import annotations

import json


def run_data(args) -> None:
    """Dispatch to the appropriate data subcommand."""
    subcmd = getattr(args, "data_command", None)
    if subcmd is None:
        print("Usage: cortexlab data {list|catalog|download|import|inspect|stats|delete}")
        return

    handlers = {
        "list": _run_list,
        "catalog": _run_catalog,
        "download": _run_download,
        "import": _run_import,
        "inspect": _run_inspect,
        "stats": _run_stats,
        "validate": _run_validate,
        "delete": _run_delete,
    }
    handler = handlers.get(subcmd)
    if handler:
        handler(args)
    else:
        print(f"Unknown data subcommand: {subcmd}")


def _run_list(args) -> None:
    """List all downloaded/imported datasets."""
    from cortexlab.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    datasets = registry.list_datasets()

    if not datasets:
        print("No datasets downloaded. Use 'cortexlab data catalog' to browse available datasets.")
        return

    print(f"{'Name':<25} {'Format':<12} {'Samples':>10} {'Source':<15}")
    print("-" * 65)
    for ds in datasets:
        name = ds.get("id", ds.get("display_name", "?"))
        fmt = ds.get("format", "?")
        samples = ds.get("num_samples", 0)
        origin = ds.get("origin", "?")
        print(f"{name:<25} {fmt:<12} {samples:>10,} {origin:<15}")


def _run_catalog(args) -> None:
    """Show the curated dataset catalog."""
    from cortexlab.data.catalog import DATASET_CATALOG
    from cortexlab.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    downloaded = {ds["id"] for ds in registry.list_datasets()}

    category_filter = getattr(args, "category", None)

    # Group by category
    categories: dict[str, list] = {}
    for profile in DATASET_CATALOG.values():
        if category_filter and profile.category != category_filter:
            continue
        categories.setdefault(profile.category, []).append(profile)

    for cat, profiles in sorted(categories.items()):
        print(f"\n  {cat.upper()}")
        print(f"  {'ID':<25} {'Name':<28} {'Samples':>10} {'Format':<12} {'Status':<10}")
        print(f"  {'-'*85}")
        for p in profiles:
            status = "downloaded" if p.id in downloaded else ""
            print(f"  {p.id:<25} {p.display_name:<28} {p.total_samples:>10,} {p.format:<12} {status:<10}")

    print(f"\n  {len(DATASET_CATALOG)} datasets available")
    print("  Download: cortexlab data download <id>")


def _run_download(args) -> None:
    """Download a dataset from the catalog."""
    from cortexlab.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    max_samples = getattr(args, "max_samples", None)

    try:
        path = registry.download(args.dataset_id, max_samples=max_samples)
        print(f"\n  Downloaded to {path}")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise


def _run_import(args) -> None:
    """Import a local JSONL file."""
    from cortexlab.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    fmt = getattr(args, "format", None)

    try:
        path = registry.import_local(
            args.file,
            name=args.name,
            format=fmt,
        )
        print(f"\n  Imported to {path}")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise


def _run_inspect(args) -> None:
    """Preview samples from a dataset."""
    from cortexlab.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    n = getattr(args, "n", 5)
    samples = registry.get_samples(args.name, n=n)

    if not samples:
        print(f"Dataset '{args.name}' not found or empty.")
        return

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i + 1} ---")
        print(json.dumps(sample, indent=2, ensure_ascii=False)[:500])


def _run_stats(args) -> None:
    """Show dataset statistics."""
    from cortexlab.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    meta = registry.get_dataset(args.name)

    if meta is None:
        print(f"Dataset '{args.name}' not found.")
        return

    print(f"\n  Dataset: {meta.get('display_name', meta.get('id', args.name))}")
    print(f"  Source:  {meta.get('source', 'unknown')}")
    print(f"  Format:  {meta.get('format', 'unknown')}")
    print(f"  License: {meta.get('license', 'unknown')}")
    print(f"  Samples: {meta.get('num_samples', 0):,}")

    if meta.get("description"):
        print(f"  Description: {meta['description']}")
    if meta.get("tags"):
        print(f"  Tags: {', '.join(meta['tags'])}")


def _run_validate(args) -> None:
    """Validate a JSONL data file."""
    from cortexlab.data.validate import validate_file

    val_path = getattr(args, "val", None)
    report = validate_file(args.file, val_path=val_path)

    print(f"\n  File: {args.file}")
    print(f"  Format: {report.format}")
    print(f"  Samples: {report.num_samples:,}")

    if report.length_stats:
        stats = report.length_stats
        print("\n  Character length stats:")
        print(f"    Min: {stats['min']:,}  Max: {stats['max']:,}  Mean: {stats['mean']:,}")
        print(f"    P50: {stats['p50']:,}  P95: {stats['p95']:,}")

    if report.num_duplicates:
        print(f"\n  Duplicates: {report.num_duplicates}")
    if report.overlap_count:
        print(f"  Train/val overlap: {report.overlap_count}")

    if report.errors:
        print(f"\n  ERRORS ({len(report.errors)}):")
        for err in report.errors[:20]:
            print(f"    - {err}")
        if len(report.errors) > 20:
            print(f"    ... and {len(report.errors) - 20} more")

    if report.warnings:
        print(f"\n  WARNINGS ({len(report.warnings)}):")
        for warn in report.warnings[:20]:
            print(f"    - {warn}")
        if len(report.warnings) > 20:
            print(f"    ... and {len(report.warnings) - 20} more")

    if report.ok and not report.warnings:
        print("\n  All checks passed!")
    elif report.ok:
        print(f"\n  Passed with {len(report.warnings)} warning(s)")
    else:
        print(f"\n  FAILED with {len(report.errors)} error(s)")


def _run_delete(args) -> None:
    """Delete a downloaded dataset."""
    from cortexlab.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    deleted = registry.delete_dataset(args.name)

    if deleted:
        print(f"  Deleted dataset '{args.name}'")
    else:
        print(f"Dataset '{args.name}' not found.")
