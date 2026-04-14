"""Model pool — multi-model lifecycle management with TTL-based eviction.

Like Ollama: models load on demand, stay warm for a configurable keep-alive
duration, then auto-unload to free memory. LRU eviction when max_models
is exceeded.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from mlx_forge.serving.model_manager import ModelManager

logger = logging.getLogger(__name__)


def parse_keep_alive(value: str | int | float | None, default: float) -> float:
    """Parse Ollama-style keep_alive values.

    Args:
        value: Keep-alive specification. Supports:
            - None → default
            - int/float → seconds
            - -1 → infinity (never unload)
            - 0 → immediate unload after request
            - "5m" → 5 minutes
            - "1h" → 1 hour
            - "30s" → 30 seconds
        default: Default value when value is None.

    Returns:
        Keep-alive duration in seconds. float('inf') for never-unload.
    """
    if value is None:
        return default

    if isinstance(value, (int, float)):
        if value < 0:
            return float("inf")
        return float(value)

    if isinstance(value, str):
        value = value.strip()
        if value == "-1":
            return float("inf")

        match = re.match(r"^(\d+(?:\.\d+)?)\s*([smhd]?)$", value)
        if match:
            num = float(match.group(1))
            unit = match.group(2) or "s"
            multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}
            return num * multiplier[unit]

        # Try as plain number
        try:
            v = float(value)
            return float("inf") if v < 0 else v
        except ValueError:
            pass

    return default


@dataclass
class ManagedModel:
    """A model with lifecycle metadata."""

    manager: ModelManager
    model_id: str
    loaded_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    keep_alive: float = 300.0  # seconds, inf = forever, 0 = immediate

    @property
    def expires_at(self) -> float | None:
        """Timestamp when this model will be evicted, or None if pinned."""
        if self.keep_alive == float("inf"):
            return None
        return self.last_access + self.keep_alive

    @property
    def is_expired(self) -> bool:
        if self.keep_alive == float("inf"):
            return False
        return time.time() > (self.last_access + self.keep_alive)

    def touch(self, keep_alive: float | None = None) -> None:
        """Update last access time and optionally keep_alive."""
        self.last_access = time.time()
        if keep_alive is not None:
            self.keep_alive = keep_alive


class ModelPool:
    """Manages multiple models with TTL-based lifecycle.

    Models load on demand via get(). Idle models are evicted after their
    keep_alive expires. LRU eviction when max_models is exceeded.

    Args:
        max_models: Maximum number of models to keep in memory.
        default_keep_alive: Default keep-alive in seconds (300 = 5 min).
    """

    def __init__(
        self,
        max_models: int = 1,
        default_keep_alive: float = 300.0,
    ):
        self._models: dict[str, ManagedModel] = {}
        self._aliases: dict[str, str] = {}
        self._max_models = max_models
        self._default_keep_alive = default_keep_alive

    def get(
        self,
        model_id: str,
        keep_alive: str | int | float | None = None,
    ) -> ModelManager:
        """Get a loaded model, loading it if necessary.

        Resolves aliases. Evicts idle models if needed to make room.

        Args:
            model_id: Model identifier (alias, HF ID, or local path).
            keep_alive: Override keep-alive for this request.

        Returns:
            The ModelManager for the requested model.
        """
        resolved_id = self.resolve_alias(model_id)
        ka = parse_keep_alive(keep_alive, self._default_keep_alive)

        # Already loaded?
        if resolved_id in self._models:
            managed = self._models[resolved_id]
            managed.touch(keep_alive=ka)
            return managed.manager

        # Need to load — make room if necessary
        self._evict_expired()
        while len(self._models) >= self._max_models:
            if not self._evict_lru():
                break  # nothing to evict

        # Load the model — resolve forge: prefix
        mgr = ModelManager()
        try:
            if resolved_id.startswith("forge:"):
                self._load_from_forge(mgr, resolved_id[6:])
            else:
                mgr.load(resolved_id)
            mgr.snapshot_base_weights()
        except Exception:
            logger.exception("Failed to load model '%s'", resolved_id)
            raise

        managed = ManagedModel(
            manager=mgr,
            model_id=resolved_id,
            keep_alive=ka,
        )
        self._models[resolved_id] = managed
        logger.info(
            "Loaded model '%s' (keep_alive=%.0fs, %d/%d slots)",
            resolved_id,
            ka,
            len(self._models),
            self._max_models,
        )
        return mgr

    def unload(self, model_id: str) -> bool:
        """Explicitly unload a model.

        Returns:
            True if the model was found and unloaded.
        """
        resolved_id = self.resolve_alias(model_id)
        if resolved_id in self._models:
            managed = self._models.pop(resolved_id)
            managed.manager.unload()
            logger.info("Unloaded model '%s'", resolved_id)
            return True
        return False

    def tick(self) -> list[str]:
        """Evict expired models. Call periodically.

        Returns:
            List of evicted model IDs.
        """
        return self._evict_expired()

    def status(self) -> list[dict]:
        """Return status of all loaded models."""
        result = []
        now = time.time()
        for mid, managed in self._models.items():
            entry = {
                "model_id": mid,
                "loaded_at": managed.loaded_at,
                "last_access": managed.last_access,
                "keep_alive": managed.keep_alive,
                "idle_seconds": round(now - managed.last_access, 1),
            }
            if managed.expires_at is not None:
                entry["expires_at"] = managed.expires_at
                entry["expires_in_seconds"] = round(
                    max(0, managed.expires_at - now), 1
                )
            else:
                entry["expires_at"] = None
                entry["expires_in_seconds"] = None
            if managed.manager.adapter_path:
                entry["adapter"] = managed.manager.adapter_path
            result.append(entry)
        return result

    @property
    def loaded_count(self) -> int:
        return len(self._models)

    @property
    def max_models(self) -> int:
        return self._max_models

    # ─── Aliases ───

    def resolve_alias(self, name: str) -> str:
        """Resolve an alias to a model ID. Returns name unchanged if not an alias."""
        return self._aliases.get(name, name)

    def add_alias(self, alias: str, model_id: str) -> None:
        """Register an alias for a model ID."""
        self._aliases[alias] = model_id

    def remove_alias(self, alias: str) -> bool:
        """Remove an alias. Returns True if it existed."""
        if alias in self._aliases:
            del self._aliases[alias]
            return True
        return False

    def list_aliases(self) -> dict[str, str]:
        """Return all aliases."""
        return dict(self._aliases)

    def load_aliases(self, path: str | Path) -> None:
        """Load aliases from a JSON file."""
        path = Path(path).expanduser()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._aliases.update(data)
                logger.info("Loaded %d aliases from %s", len(data), path)

    def save_aliases(self, path: str | Path) -> None:
        """Save aliases to a JSON file."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._aliases, f, indent=2)

    # ─── Internal ───

    def _load_from_forge(self, mgr: ModelManager, forge_name: str) -> None:
        """Load a model from a forge spec."""
        from mlx_forge.forge import get_forge

        forge = get_forge(forge_name)
        if forge is None:
            raise FileNotFoundError(f"Forge '{forge_name}' not found")

        load_args = forge.to_load_args()
        from mlx_forge.inference.engine import load_for_inference

        model, tokenizer = load_for_inference(
            load_args["model_path"],
            adapter_path=load_args.get("adapter_path"),
        )
        mgr._model = model
        mgr._tokenizer = tokenizer
        mgr._model_id = f"forge:{forge_name}"
        logger.info("Loaded forge '%s' (base=%s)", forge_name, forge.base)

    def _evict_expired(self) -> list[str]:
        """Remove models past their keep_alive. Returns evicted IDs."""
        expired = [mid for mid, m in self._models.items() if m.is_expired]
        for mid in expired:
            managed = self._models.pop(mid)
            managed.manager.unload()
            logger.info("Evicted expired model '%s'", mid)
        return expired

    def _evict_lru(self) -> bool:
        """Evict the least-recently-used model. Returns True if one was evicted."""
        if not self._models:
            return False

        # Don't evict pinned models (keep_alive=inf) unless no other choice
        candidates = [
            (mid, m)
            for mid, m in self._models.items()
            if m.keep_alive != float("inf")
        ]
        if not candidates:
            candidates = list(self._models.items())

        lru_id = min(candidates, key=lambda x: x[1].last_access)[0]
        managed = self._models.pop(lru_id)
        managed.manager.unload()
        logger.info("LRU-evicted model '%s'", lru_id)
        return True
