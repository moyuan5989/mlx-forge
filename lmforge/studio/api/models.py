"""Models API — model discovery from HF cache."""

from __future__ import annotations

from fastapi import APIRouter

from lmforge.studio.services.model_service import ModelService

router = APIRouter(prefix="/api/v1/models", tags=["models"])

_model_service = ModelService()


def get_model_service() -> ModelService:
    return _model_service


def set_model_service(service: ModelService):
    global _model_service
    _model_service = service


@router.get("")
def list_models():
    """List all downloaded models from the HF cache."""
    return get_model_service().list_models()


@router.get("/supported")
def list_supported():
    """List supported model architectures."""
    return get_model_service().list_supported_architectures()


# ── V2: Model Library ─────────────────────────────────────────────────────────

from lmforge.studio.services.model_library_service import ModelLibraryService

router_v2 = APIRouter(prefix="/api/v2/models", tags=["models-v2"])

_library_service = ModelLibraryService()


@router_v2.get("/library")
def list_library():
    """List curated model catalog with memory estimates and download status."""
    return _library_service.list_library()
