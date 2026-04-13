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
        self._draft_model = None
        self._draft_model_id: Optional[str] = None
        self._base_weights: Optional[dict] = None  # snapshot for adapter hot-swap
        self._adapter_path: Optional[str] = None

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
    def draft_model(self):
        return self._draft_model

    @property
    def draft_model_id(self) -> Optional[str]:
        return self._draft_model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def has_draft(self) -> bool:
        return self._draft_model is not None

    @property
    def adapter_path(self) -> Optional[str]:
        return self._adapter_path

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

    def load_draft(self, draft_model_id: str) -> None:
        """Load a draft model for speculative decoding.

        Args:
            draft_model_id: HF model ID or local path for draft model
        """
        from mlx_forge.inference.engine import load_for_inference

        self._draft_model, _ = load_for_inference(draft_model_id)
        self._draft_model_id = draft_model_id

    def snapshot_base_weights(self) -> None:
        """Take a snapshot of base model weights for adapter hot-swap.

        Call after loading a base model (before any adapter is applied).
        """
        if self._model is None:
            return
        import mlx.core as mx

        self._base_weights = {
            k: mx.array(v) for k, v in self._model.parameters().items()
            if not k.startswith("_")
        }

    def load_adapter(self, adapter_path: str) -> None:
        """Hot-swap a LoRA adapter: restore base weights, then apply new adapter.

        Args:
            adapter_path: Path to directory containing adapters.safetensors.

        Raises:
            ValueError: If no model is loaded or no base weight snapshot exists.
            FileNotFoundError: If adapter file doesn't exist.
        """
        import mlx.core as mx

        if self._model is None:
            raise ValueError("No model loaded — load a model before applying adapters")

        adapter_dir = Path(adapter_path).expanduser()
        adapter_file = adapter_dir / "adapters.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(
                f"No adapters.safetensors in {adapter_dir}"
            )

        # Restore base weights first (if we have a snapshot)
        if self._base_weights is not None:
            self._model.load_weights(list(self._base_weights.items()), strict=False)

        # Apply new adapter
        adapter_weights = mx.load(str(adapter_file))
        self._model.load_weights(list(adapter_weights.items()), strict=False)
        mx.eval(self._model.parameters())
        self._adapter_path = str(adapter_dir)

    def unload_adapter(self) -> None:
        """Remove current adapter, restoring base weights.

        Raises:
            ValueError: If no base weight snapshot exists.
        """
        if self._base_weights is None:
            raise ValueError("No base weight snapshot — cannot restore base weights")

        import mlx.core as mx

        self._model.load_weights(list(self._base_weights.items()), strict=False)
        mx.eval(self._model.parameters())
        self._adapter_path = None

    def adapter_info(self) -> dict:
        """Return information about the currently loaded adapter."""
        return {
            "adapter_loaded": self._adapter_path is not None,
            "adapter_path": self._adapter_path,
            "model_id": self._model_id,
        }

    def unload(self) -> None:
        """Free current model from memory."""
        self._model = None
        self._tokenizer = None
        self._model_id = None
        self._draft_model = None
        self._draft_model_id = None
        self._base_weights = None
        self._adapter_path = None

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
