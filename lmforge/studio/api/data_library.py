"""Data Library API — catalog browsing and dataset management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lmforge.studio.services.data_library_service import DataLibraryService

router = APIRouter(prefix="/api/v2/data", tags=["data-library"])

_service = DataLibraryService()


def get_service() -> DataLibraryService:
    return _service


def set_service(service: DataLibraryService):
    global _service
    _service = service


class DownloadRequest(BaseModel):
    catalog_id: str
    max_samples: int | None = None


@router.get("/catalog")
def list_catalog():
    """List curated dataset catalog with download status."""
    return get_service().list_catalog()


@router.get("/datasets")
def list_datasets():
    """List all downloaded/imported datasets."""
    return get_service().list_downloaded()


@router.post("/download")
def download_dataset(req: DownloadRequest):
    """Download a dataset from the catalog."""
    try:
        return get_service().download(req.catalog_id, max_samples=req.max_samples)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{name}")
def get_dataset(name: str):
    """Get metadata and stats for a specific dataset."""
    ds = get_service().get_dataset(name)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return ds


@router.get("/datasets/{name}/samples")
def get_dataset_samples(name: str, n: int = 5):
    """Preview samples from a dataset."""
    samples = get_service().get_samples(name, n=n)
    if not samples:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found or empty")
    return samples


@router.delete("/datasets/{name}")
def delete_dataset(name: str):
    """Delete a downloaded dataset."""
    deleted = get_service().delete(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return {"status": "deleted", "name": name}
