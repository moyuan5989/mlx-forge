"""Dataset cache discovery and management.

V2: Scans ~/.lmforge/datasets/processed/*/meta.json for tokenized datasets.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

class DatasetService:
    """Discovers and manages processed (tokenized) datasets."""

    def __init__(self, datasets_dir: str = "~/.lmforge/datasets"):
        self.datasets_dir = Path(datasets_dir).expanduser()

    def list_datasets(self) -> list[dict]:
        """List all processed (tokenized) datasets."""
        processed_dir = self.datasets_dir / "processed"
        if not processed_dir.exists():
            return []

        results = []
        for entry in sorted(processed_dir.iterdir()):
            if not entry.is_dir():
                continue
            meta_path = entry / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["name"] = entry.name
                meta["path"] = str(entry)
                results.append(meta)
            except Exception:
                continue
        return results

    def get_dataset(self, name: str) -> Optional[dict]:
        """Get metadata for a specific processed dataset.

        Args:
            name: The dataset directory name (e.g., "train--Qwen--Qwen3-0.6B").
        """
        processed_dir = self.datasets_dir / "processed"
        if not processed_dir.exists():
            return None

        entry = processed_dir / name
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta["name"] = name
            meta["path"] = str(entry)
            return meta
        except Exception:
            return None

    def delete_dataset(self, name: str) -> bool:
        """Delete a processed dataset. Returns True if deleted."""
        entry = self.datasets_dir / "processed" / name
        if not entry.exists():
            return False
        shutil.rmtree(entry)
        return True
