"""Dataset cache discovery and management.

Scans ~/.lmforge/cache/preprocessed/*/meta.json for cached datasets.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional


class DatasetService:
    """Discovers and manages preprocessed dataset caches."""

    def __init__(self, cache_dir: str = "~/.lmforge/cache/preprocessed"):
        self.cache_dir = Path(cache_dir).expanduser()

    def list_datasets(self) -> list[dict]:
        """List all cached datasets.

        Returns list of dicts with dataset metadata from meta.json.
        """
        if not self.cache_dir.exists():
            return []

        datasets = []
        for entry in sorted(self.cache_dir.iterdir()):
            if not entry.is_dir():
                continue
            meta_path = entry / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["fingerprint"] = entry.name
                meta["path"] = str(entry)
                datasets.append(meta)
            except Exception:
                continue
        return datasets

    def get_dataset(self, fingerprint: str) -> Optional[dict]:
        """Get metadata for a specific cached dataset.

        Args:
            fingerprint: The dataset fingerprint (directory name).

        Returns:
            Dict with meta.json contents + fingerprint, or None.
        """
        cache_path = self.cache_dir / fingerprint
        meta_path = cache_path / "meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta["fingerprint"] = fingerprint
            meta["path"] = str(cache_path)
            return meta
        except Exception:
            return None

    def delete_dataset(self, fingerprint: str) -> bool:
        """Delete a cached dataset. Returns True if deleted."""
        cache_path = self.cache_dir / fingerprint
        if not cache_path.exists():
            return False
        shutil.rmtree(cache_path)
        return True
