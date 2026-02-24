"""Datasets API — preprocessed dataset cache management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lmforge.studio.services.dataset_service import DatasetService

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

_dataset_service = DatasetService()


def get_dataset_service() -> DatasetService:
    return _dataset_service


def set_dataset_service(service: DatasetService):
    global _dataset_service
    _dataset_service = service


@router.get("")
def list_datasets():
    """List all cached preprocessed datasets."""
    return get_dataset_service().list_datasets()


@router.get("/{fingerprint}")
def get_dataset(fingerprint: str):
    """Get metadata for a cached dataset."""
    ds = get_dataset_service().get_dataset(fingerprint)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{fingerprint}' not found")
    return ds


@router.delete("/{fingerprint}")
def delete_dataset(fingerprint: str):
    """Delete a cached dataset."""
    deleted = get_dataset_service().delete_dataset(fingerprint)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Dataset '{fingerprint}' not found")
    return {"status": "deleted", "fingerprint": fingerprint}
