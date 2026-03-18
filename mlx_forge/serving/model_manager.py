"""Model manager — loads and caches a single model in memory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class ModelManager:
    """Manages a single loaded model for serving.

    Resolution chain:
    1. Run ID -> ~/.mlxforge/runs/{id}/ (load base + adapter)
    2. Export name -> ~/.mlxforge/exports/{id}/
    3. HF repo ID -> download via resolve_model
    4. Local path
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_id: Optional[str] = None

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_id: str, adapter: str | None = None) -> None:
        """Load a model. Unloads previous model first.

        Resolution: run_id -> export -> HF repo -> local path.
        """
        # Unload current model
        self.unload()

        runs_dir = Path("~/.mlxforge/runs").expanduser()
        exports_dir = Path("~/.mlxforge/exports").expanduser()

        adapter_path = adapter

        # 1. Check if it's a run ID
        run_dir = runs_dir / model_id
        if run_dir.exists():
            import yaml

            config_path = run_dir / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                base_model = config.get("model", {}).get("path", model_id)
                # Find best checkpoint
                ckpt_dir = run_dir / "checkpoints"
                best_link = ckpt_dir / "best"
                if best_link.exists():
                    adapter_path = str(best_link.resolve())
                else:
                    ckpts = sorted(
                        [d for d in ckpt_dir.iterdir() if d.is_dir() and not d.is_symlink()]
                    )
                    if ckpts:
                        adapter_path = str(ckpts[-1])
                from mlx_forge.inference.engine import load_for_inference

                self._model, self._tokenizer = load_for_inference(
                    base_model,
                    adapter_path=adapter_path,
                    trust_remote_code=config.get("model", {}).get("trust_remote_code", False),
                )
                self._model_id = model_id
                return

        # 2. Check if it's an export
        export_dir = exports_dir / model_id
        if export_dir.exists() and (export_dir / "model.safetensors").exists():
            from mlx_forge.inference.engine import load_for_inference

            self._model, self._tokenizer = load_for_inference(str(export_dir))
            self._model_id = model_id
            return

        # 3. Try as HF repo or local path
        from mlx_forge.inference.engine import load_for_inference

        self._model, self._tokenizer = load_for_inference(
            model_id,
            adapter_path=adapter_path,
        )
        self._model_id = model_id

    def unload(self) -> None:
        """Free current model from memory."""
        self._model = None
        self._tokenizer = None
        self._model_id = None

    def list_available(self) -> list[dict]:
        """List available models from runs, exports, and HF cache."""
        models = []

        # Scan runs
        runs_dir = Path("~/.mlxforge/runs").expanduser()
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.iterdir()):
                if run_dir.is_dir():
                    models.append(
                        {
                            "id": run_dir.name,
                            "source": "run",
                            "path": str(run_dir),
                        }
                    )

        # Scan exports
        exports_dir = Path("~/.mlxforge/exports").expanduser()
        if exports_dir.exists():
            for export_dir in sorted(exports_dir.iterdir()):
                if export_dir.is_dir() and (export_dir / "model.safetensors").exists():
                    models.append(
                        {
                            "id": export_dir.name,
                            "source": "export",
                            "path": str(export_dir),
                        }
                    )

        return models
