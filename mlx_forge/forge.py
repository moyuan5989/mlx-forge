"""Forge files — bundled model definitions for train-to-serve workflows.

A ForgeSpec bundles: base model + adapter + system prompt + default parameters.
Stored as YAML in ~/.mlxforge/forges/{name}.yaml.

Example:
    name: my-assistant
    base: Qwen/Qwen3-0.6B
    adapter: "run:sft-2026-04-12"
    system: |
      You are a helpful coding assistant.
    parameters:
      temperature: 0.7
      top_k: 40
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

FORGES_DIR = Path("~/.mlxforge/forges").expanduser()
RUNS_DIR = Path("~/.mlxforge/runs").expanduser()


@dataclass
class ForgeSpec:
    """A bundled model definition: base + adapter + system prompt + params."""

    name: str
    base: str  # HF ID or local path
    adapter: str | None = None  # "run:{id}", path, or None
    system: str | None = None  # default system prompt
    parameters: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ForgeSpec:
        """Load a ForgeSpec from a YAML file.

        Args:
            path: Path to the forge YAML file.

        Returns:
            Parsed ForgeSpec.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If required fields are missing.
        """
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Forge file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid forge file: {path}")
        if "base" not in data:
            raise ValueError(f"Forge file missing 'base' field: {path}")

        return cls(
            name=data.get("name", path.stem),
            base=data["base"],
            adapter=data.get("adapter"),
            system=data.get("system"),
            parameters=data.get("parameters", {}),
        )

    @classmethod
    def from_run(cls, run_id: str) -> ForgeSpec:
        """Create a ForgeSpec from a training run.

        Reads the run's config.yaml to find the base model and uses the
        best checkpoint as the adapter.

        Args:
            run_id: Training run ID (directory name in ~/.mlxforge/runs/).

        Returns:
            ForgeSpec pointing to the run's base model + best adapter.

        Raises:
            FileNotFoundError: If the run doesn't exist.
        """
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_dir}")

        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Run config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        base_model = config.get("model", {}).get("path", "")
        if not base_model:
            raise ValueError(f"Run config missing model.path: {config_path}")

        # Find best checkpoint
        ckpt_dir = run_dir / "checkpoints"
        adapter = None
        if ckpt_dir.exists():
            best_link = ckpt_dir / "best"
            if best_link.exists():
                adapter = str(best_link.resolve())
            else:
                ckpts = sorted(
                    [d for d in ckpt_dir.iterdir() if d.is_dir() and not d.is_symlink()]
                )
                if ckpts:
                    adapter = str(ckpts[-1])

        return cls(
            name=run_id,
            base=base_model,
            adapter=adapter,
            parameters=config.get("training", {}).get("parameters", {}),
        )

    def save(self, path: str | Path | None = None) -> Path:
        """Save the ForgeSpec to a YAML file.

        Args:
            path: Output path. Defaults to ~/.mlxforge/forges/{name}.yaml.

        Returns:
            Path to the saved file.
        """
        if path is None:
            path = FORGES_DIR / f"{self.name}.yaml"
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "base": self.base,
        }
        if self.adapter:
            data["adapter"] = self.adapter
        if self.system:
            data["system"] = self.system
        if self.parameters:
            data["parameters"] = self.parameters

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return path

    def resolve_adapter_path(self) -> str | None:
        """Resolve adapter reference to an actual file path.

        Handles:
        - "run:{id}" → best checkpoint from training run
        - Absolute/relative path → used as-is
        - None → no adapter

        Returns:
            Resolved adapter directory path, or None.
        """
        if self.adapter is None:
            return None

        if self.adapter.startswith("run:"):
            run_id = self.adapter[4:]
            run_dir = RUNS_DIR / run_id
            if not run_dir.exists():
                raise FileNotFoundError(f"Run not found: {run_dir}")
            ckpt_dir = run_dir / "checkpoints"
            best = ckpt_dir / "best"
            if best.exists():
                return str(best.resolve())
            ckpts = sorted(
                [d for d in ckpt_dir.iterdir() if d.is_dir() and not d.is_symlink()]
            )
            if ckpts:
                return str(ckpts[-1])
            raise FileNotFoundError(f"No checkpoints in run: {run_id}")

        return self.adapter

    def to_load_args(self) -> dict:
        """Convert to arguments for ModelManager.load() or load_for_inference().

        Returns:
            Dict with 'model_path' and optionally 'adapter_path'.
        """
        result = {"model_path": self.base}
        adapter = self.resolve_adapter_path()
        if adapter:
            result["adapter_path"] = adapter
        return result


def list_forges() -> list[ForgeSpec]:
    """List all saved forge specs.

    Returns:
        List of ForgeSpec objects from ~/.mlxforge/forges/.
    """
    if not FORGES_DIR.exists():
        return []

    forges = []
    for path in sorted(FORGES_DIR.glob("*.yaml")):
        try:
            forges.append(ForgeSpec.from_yaml(path))
        except Exception as e:
            logger.warning("Skipping invalid forge %s: %s", path.name, e)

    return forges


def get_forge(name: str) -> ForgeSpec | None:
    """Get a forge by name.

    Args:
        name: Forge name (without .yaml extension).

    Returns:
        ForgeSpec or None if not found.
    """
    path = FORGES_DIR / f"{name}.yaml"
    if path.exists():
        return ForgeSpec.from_yaml(path)
    return None


def delete_forge(name: str) -> bool:
    """Delete a forge by name.

    Returns:
        True if the forge was found and deleted.
    """
    path = FORGES_DIR / f"{name}.yaml"
    if path.exists():
        path.unlink()
        return True
    return False
