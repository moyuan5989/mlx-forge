"""Run manifest and environment info for CortexLab v0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HardwareInfo:
    chip: str
    memory_gb: int
    gpu_cores: int
    os: str


@dataclass
class RunManifest:
    schema_version: int
    config: dict
    cortexlab_version: str
    mlx_version: str
    python_version: str
    hardware: HardwareInfo
    data_fingerprint: str
    created_at: str
    model_resolution: dict  # Model resolution metadata (source_id, revision, paths)


@dataclass
class EnvironmentInfo:
    python_version: str
    mlx_version: str
    cortexlab_version: str
    platform: str
    os_version: str
    chip: str
    memory_gb: int
    gpu_cores: int


def collect_environment() -> EnvironmentInfo:
    """Collect current environment information."""
    import os
    import platform
    import sys

    import mlx.core as mx

    from cortexlab._version import __version__

    # Get MLX version
    mlx_version = mx.__version__

    # Get platform info
    platform_str = platform.platform()
    os_version = platform.version()

    # Get chip and memory info
    chip = platform.machine()
    memory_gb = 0
    gpu_cores = 0

    if mx.metal.is_available():
        device_info = mx.metal.device_info()
        chip = device_info.get("device_name", chip)
        gpu_cores = device_info.get("gpu_cores", 0)

        # Try to get memory info
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            num_pages = os.sysconf("SC_PHYS_PAGES")
            memory_gb = int((page_size * num_pages) / 1e9)
        except (ValueError, OSError):
            memory_gb = 0

    return EnvironmentInfo(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        mlx_version=mlx_version,
        cortexlab_version=__version__,
        platform=platform_str,
        os_version=os_version,
        chip=chip,
        memory_gb=memory_gb,
        gpu_cores=gpu_cores,
    )


def write_manifest(
    run_dir: Path,
    config: dict,
    data_fingerprint: str,
    model_resolution: dict,
) -> RunManifest:
    """Write manifest.json and environment.json to the run directory."""
    import json
    from datetime import datetime, timezone

    import mlx.core as mx

    from cortexlab._version import __version__

    # Collect environment info
    env_info = collect_environment()

    # Create hardware info
    hardware = HardwareInfo(
        chip=env_info.chip,
        memory_gb=env_info.memory_gb,
        gpu_cores=env_info.gpu_cores,
        os=env_info.os_version,
    )

    # Create manifest
    manifest = RunManifest(
        schema_version=1,
        config=config,
        cortexlab_version=__version__,
        mlx_version=mx.__version__,
        python_version=env_info.python_version,
        hardware=hardware,
        data_fingerprint=data_fingerprint,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        model_resolution=model_resolution,
    )

    # Write manifest.json
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_dict = {
        "schema_version": manifest.schema_version,
        "config": manifest.config,
        "cortexlab_version": manifest.cortexlab_version,
        "mlx_version": manifest.mlx_version,
        "python_version": manifest.python_version,
        "hardware": {
            "chip": manifest.hardware.chip,
            "memory_gb": manifest.hardware.memory_gb,
            "gpu_cores": manifest.hardware.gpu_cores,
            "os": manifest.hardware.os,
        },
        "data_fingerprint": manifest.data_fingerprint,
        "created_at": manifest.created_at,
        "model_resolution": manifest.model_resolution,
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest_dict, indent=2))

    # Write environment.json
    env_dict = {
        "python_version": env_info.python_version,
        "mlx_version": env_info.mlx_version,
        "cortexlab_version": env_info.cortexlab_version,
        "platform": env_info.platform,
        "os_version": env_info.os_version,
        "chip": env_info.chip,
        "memory_gb": env_info.memory_gb,
        "gpu_cores": env_info.gpu_cores,
    }

    (run_dir / "environment.json").write_text(json.dumps(env_dict, indent=2))

    return manifest
