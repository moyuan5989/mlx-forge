"""Model discovery from HuggingFace cache.

Scans ~/.cache/huggingface/hub/models--*/snapshots/ for downloaded models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class ModelService:
    """Discovers downloaded models from the HF cache."""

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            # Standard HF cache location
            self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        else:
            self.cache_dir = Path(cache_dir).expanduser()

    def list_models(self) -> list[dict]:
        """List all downloaded models in the HF cache.

        Returns list of dicts with: id, path, architecture, supported, size_gb.
        """
        if not self.cache_dir.exists():
            return []

        models = []
        for model_dir in sorted(self.cache_dir.iterdir()):
            if not model_dir.name.startswith("models--"):
                continue
            try:
                info = self._scan_model(model_dir)
                if info:
                    models.append(info)
            except Exception:
                continue
        return models

    def get_model(self, model_id: str) -> Optional[dict]:
        """Get details for a specific model by HF ID (e.g., 'Qwen/Qwen3-0.8B').

        Returns model info dict or None.
        """
        # Convert model_id to cache dir name: Qwen/Qwen3-0.8B -> models--Qwen--Qwen3-0.8B
        cache_name = "models--" + model_id.replace("/", "--")
        model_dir = self.cache_dir / cache_name
        if not model_dir.exists():
            return None
        return self._scan_model(model_dir)

    def _scan_model(self, model_dir: Path) -> Optional[dict]:
        """Scan a single model cache directory."""
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        # Get the latest snapshot (by directory listing)
        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshots:
            return None

        # Use latest snapshot
        snapshot = max(snapshots, key=lambda d: d.stat().st_mtime)

        # Parse model ID from directory name: models--Qwen--Qwen3-0.8B -> Qwen/Qwen3-0.8B
        parts = model_dir.name.removeprefix("models--").split("--")
        model_id = "/".join(parts)

        info = {
            "id": model_id,
            "path": str(snapshot),
            "architecture": None,
            "supported": False,
            "size_gb": None,
        }

        # Read config.json for architecture info
        config_path = snapshot / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                model_type = config.get("model_type")
                info["architecture"] = model_type
                # Check if supported by LMForge
                from lmforge.models.registry import is_supported
                info["supported"] = is_supported(model_type) if model_type else False
            except Exception:
                pass

        # Estimate model size from safetensors files
        total_bytes = 0
        for sf in snapshot.glob("*.safetensors"):
            total_bytes += sf.stat().st_size
        if total_bytes > 0:
            info["size_gb"] = round(total_bytes / (1024**3), 2)

        return info

    def list_supported_architectures(self) -> list[str]:
        """Return list of supported architecture names."""
        from lmforge.models.registry import list_supported_architectures
        return list_supported_architectures()
