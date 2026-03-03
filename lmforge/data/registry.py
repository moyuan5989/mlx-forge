"""Dataset download, import, and management for LMForge.

Manages the ~/.lmforge/datasets/ directory structure:
  raw/{dataset_id}/data.jsonl   — converted raw samples
  raw/{dataset_id}/meta.json    — dataset metadata
  processed/{name}--{model}/    — tokenized Arrow datasets
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from lmforge.data.catalog import DATASET_CATALOG, DatasetProfile
from lmforge.data.converter import convert_dataset


class DatasetRegistry:
    """Manages downloaded and imported datasets."""

    def __init__(self, base_dir: str = "~/.lmforge/datasets"):
        self.base_dir = Path(base_dir).expanduser()

    def download(
        self,
        catalog_id: str,
        *,
        max_samples: int | None = None,
    ) -> Path:
        """Download a catalog dataset, convert to standard format, save as JSONL.

        Args:
            catalog_id: ID from the dataset catalog.
            max_samples: Limit number of samples (useful for Apple Silicon).

        Returns:
            Path to the raw dataset directory.
        """
        if catalog_id not in DATASET_CATALOG:
            raise ValueError(
                f"Unknown catalog dataset: '{catalog_id}'. "
                f"Available: {', '.join(sorted(DATASET_CATALOG.keys()))}"
            )

        profile = DATASET_CATALOG[catalog_id]
        raw_path = self.base_dir / "raw" / catalog_id

        # Check if already downloaded
        if (raw_path / "data.jsonl").exists():
            meta = self._load_meta(raw_path)
            if meta:
                return raw_path

        from datasets import load_dataset

        print(f"Downloading {profile.display_name} from {profile.source}...")
        ds = load_dataset(
            profile.source,
            split=profile.split,
            name=profile.subset,
        )

        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
            print(f"  Sampled {max_samples} of {profile.total_samples} total")

        print(f"Converting to LMForge format ({profile.format})...")
        samples = convert_dataset(ds, profile)

        # Save as standard JSONL
        raw_path.mkdir(parents=True, exist_ok=True)
        data_path = raw_path / "data.jsonl"
        with open(data_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Save metadata
        meta = {
            "id": profile.id,
            "source": profile.source,
            "display_name": profile.display_name,
            "category": profile.category,
            "format": profile.format,
            "description": profile.description,
            "license": profile.license,
            "tags": profile.tags,
            "num_samples": len(samples),
            "origin": "catalog",
        }
        with open(raw_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Saved {len(samples)} samples to {raw_path}")
        return raw_path

    def import_local(
        self,
        path: str,
        *,
        name: str,
        format: str | None = None,
    ) -> Path:
        """Import a local JSONL file into the registry.

        Args:
            path: Path to the JSONL file.
            name: Name for the dataset in the registry.
            format: Optional format override ('chat', 'completions', 'text', 'preference').

        Returns:
            Path to the raw dataset directory.
        """
        from lmforge.data.formats import detect_format, validate_samples

        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Read and validate
        with open(src) as f:
            samples = [json.loads(line) for line in f if line.strip()]

        if not samples:
            raise ValueError(f"No samples found in {path}")

        fmt = format or detect_format(samples)
        errors = validate_samples(samples, fmt)
        if errors:
            raise ValueError(f"Validation failed: {errors[0]}")

        # Copy to registry
        raw_path = self.base_dir / "raw" / name
        raw_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, raw_path / "data.jsonl")

        meta = {
            "id": name,
            "source": str(src.resolve()),
            "display_name": name,
            "category": "custom",
            "format": fmt,
            "description": f"Imported from {src.name}",
            "license": "unknown",
            "tags": ["custom"],
            "num_samples": len(samples),
            "origin": "local",
        }
        with open(raw_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return raw_path

    def list_datasets(self) -> list[dict]:
        """List all downloaded/imported datasets."""
        raw_dir = self.base_dir / "raw"
        if not raw_dir.exists():
            return []

        results = []
        for entry in sorted(raw_dir.iterdir()):
            if not entry.is_dir():
                continue
            meta = self._load_meta(entry)
            if meta:
                meta["path"] = str(entry)
                meta["downloaded"] = True
                results.append(meta)
        return results

    def get_dataset(self, name: str) -> dict | None:
        """Get metadata for a specific dataset."""
        raw_path = self.base_dir / "raw" / name
        meta = self._load_meta(raw_path)
        if meta:
            meta["path"] = str(raw_path)
            meta["downloaded"] = True
        return meta

    def get_data_path(self, name: str) -> Path | None:
        """Get the JSONL data file path for a dataset."""
        data_path = self.base_dir / "raw" / name / "data.jsonl"
        return data_path if data_path.exists() else None

    def get_samples(self, name: str, n: int = 5) -> list[dict]:
        """Preview samples from a dataset."""
        data_path = self.get_data_path(name)
        if data_path is None:
            return []

        samples = []
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def delete_dataset(self, name: str) -> bool:
        """Delete a downloaded dataset. Returns True if deleted."""
        raw_path = self.base_dir / "raw" / name
        if not raw_path.exists():
            return False
        shutil.rmtree(raw_path)
        return True

    def list_catalog(self) -> list[DatasetProfile]:
        """Return the full curated catalog."""
        return list(DATASET_CATALOG.values())

    def _load_meta(self, path: Path) -> dict | None:
        """Load meta.json from a dataset directory."""
        meta_path = path / "meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return None
