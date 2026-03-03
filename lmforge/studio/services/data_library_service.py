"""Data library service for Studio — catalog browsing and dataset management."""

from __future__ import annotations

from lmforge.data.catalog import DATASET_CATALOG
from lmforge.data.registry import DatasetRegistry


class DataLibraryService:
    """Service for browsing the catalog and managing datasets."""

    def __init__(self, base_dir: str = "~/.lmforge/datasets"):
        self._registry = DatasetRegistry(base_dir)

    def list_catalog(self) -> list[dict]:
        """Return curated catalog with download status."""
        downloaded = {ds["id"] for ds in self._registry.list_datasets()}

        results = []
        for profile in DATASET_CATALOG.values():
            d = profile.to_dict()
            d["downloaded"] = profile.id in downloaded
            results.append(d)
        return results

    def list_downloaded(self) -> list[dict]:
        """Return downloaded/imported datasets with stats."""
        return self._registry.list_datasets()

    def download(self, catalog_id: str, max_samples: int | None = None) -> dict:
        """Download a catalog dataset."""
        path = self._registry.download(catalog_id, max_samples=max_samples)
        meta = self._registry.get_dataset(catalog_id)
        return meta or {"id": catalog_id, "path": str(path), "status": "downloaded"}

    def get_dataset(self, name: str) -> dict | None:
        """Get metadata for a specific dataset."""
        return self._registry.get_dataset(name)

    def get_samples(self, name: str, n: int = 5) -> list[dict]:
        """Preview samples for inspection."""
        return self._registry.get_samples(name, n=n)

    def delete(self, name: str) -> bool:
        """Delete a downloaded dataset."""
        return self._registry.delete_dataset(name)
