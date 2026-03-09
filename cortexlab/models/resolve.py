"""Model resolution layer for CortexLab v0.

Resolves Hugging Face model IDs to local paths, handling caching, offline mode,
and revision pinning. Ensures all network access happens before training begins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ResolvedModel:
    """Result of model resolution.

    Attributes:
        source_id: Original HF repo ID or local path
        resolved_revision: Pinned commit hash (None for local paths)
        local_path: Absolute path to model directory (HF snapshot or local dir)
        is_local: True if source was a local path (no HF resolution)
        resolution_metadata: Dict for manifest.json
    """
    source_id: str
    resolved_revision: Optional[str]
    local_path: str
    is_local: bool
    resolution_metadata: dict


def is_hf_repo_id(path: str) -> bool:
    """Check if a path looks like a Hugging Face repo ID.

    HF repo IDs have format: "org/model" or "org/model/subpath"
    Local paths exist on disk or are absolute/relative paths.

    Args:
        path: Model path string

    Returns:
        True if path appears to be an HF repo ID
    """
    # Check if it's an existing local path
    expanded = Path(path).expanduser()
    if expanded.exists():
        return False

    # Check if it looks like an absolute or relative path
    if os.path.isabs(path) or path.startswith("./") or path.startswith("../"):
        return False

    # Check if it has the org/model format
    parts = path.split("/")
    if len(parts) >= 2 and not path.startswith("/"):
        # Looks like org/model
        return True

    return False


def resolve_model(
    model_path: str,
    *,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
) -> ResolvedModel:
    """Resolve a model path to a local directory.

    For local paths: validates existence and returns as-is.
    For HF repo IDs: resolves revision, downloads to HF cache if needed.

    Args:
        model_path: HF repo ID (e.g., "Qwen/Qwen3-0.8B") or local path
        revision: Optional HF revision/commit hash to pin to
        trust_remote_code: Whether to trust remote code (passed to HF)
        token: Optional HF token for gated models

    Returns:
        ResolvedModel with local path and resolution metadata

    Raises:
        FileNotFoundError: Local path doesn't exist
        ValueError: HF repo not found or network error
        PermissionError: Gated model requires authentication
    """
    # Check if it's a local path
    if not is_hf_repo_id(model_path):
        local_path = Path(model_path).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(
                f"Local model path does not exist: {model_path}\n"
                f"Resolved to: {local_path}"
            )

        return ResolvedModel(
            source_id=model_path,
            resolved_revision=None,
            local_path=str(local_path),
            is_local=True,
            resolution_metadata={
                "source_id": model_path,
                "local_path": str(local_path),
                "is_local": True,
            },
        )

    # It's an HF repo ID - resolve it
    try:
        from huggingface_hub import model_info, snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models from Hugging Face Hub.\n"
            "It should be installed as a dependency of transformers.\n"
            "Try: pip install huggingface_hub"
        )

    # Check offline mode
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    # Resolve revision if not provided
    final_revision = revision
    if not offline and final_revision is None:
        try:
            print(f"Resolving model: {model_path}")
            info = model_info(model_path, token=token)
            final_revision = info.sha
            print(f"  → Latest revision: {final_revision[:8]}")
        except GatedRepoError:
            raise PermissionError(
                f"Model '{model_path}' requires authentication.\n\n"
                f"This is a gated model. To access it:\n"
                f"  1. Accept the license at https://huggingface.co/{model_path}\n"
                f"  2. Set your HF token:  huggingface-cli login\n"
                f"  3. Or set the environment variable:  export HF_TOKEN=hf_...\n\n"
                f"Then retry your command."
            )
        except RepositoryNotFoundError:
            raise ValueError(
                f"Model '{model_path}' not found on Hugging Face Hub.\n\n"
                f"Check the model ID at https://huggingface.co/models"
            )
        except Exception as e:
            if "connection" in str(e).lower() or "network" in str(e).lower():
                raise ValueError(
                    f"Cannot resolve model '{model_path}': network unavailable.\n\n"
                    f"If you have previously downloaded this model, try offline mode:\n"
                    f"  export HF_HUB_OFFLINE=1\n\n"
                    f"Or use a local model path:\n"
                    f"  model:\n"
                    f"    path: \"/path/to/local/model\""
                )
            raise

    # Download/retrieve from cache
    try:
        if offline:
            print(f"Resolving model: {model_path} (offline mode)")

        local_path = snapshot_download(
            repo_id=model_path,
            revision=final_revision,
            token=token,
            local_files_only=offline,
        )

        if offline:
            print("  → Using cached snapshot")
        else:
            print(f"  → Downloaded to: {local_path}")

        return ResolvedModel(
            source_id=model_path,
            resolved_revision=final_revision,
            local_path=local_path,
            is_local=False,
            resolution_metadata={
                "source_id": model_path,
                "resolved_revision": final_revision,
                "local_path": local_path,
                "is_local": False,
            },
        )

    except FileNotFoundError:
        if offline:
            raise ValueError(
                f"Model '{model_path}' not found in local cache (offline mode).\n\n"
                f"Download it first:\n"
                f"  HF_HUB_OFFLINE=0 cortexlab train --config train.yaml\n"
                f"  # or\n"
                f"  huggingface-cli download {model_path}"
            )
        raise
    except OSError as e:
        if "no space left" in str(e).lower() or "disk" in str(e).lower():
            raise OSError(
                f"Insufficient disk space to download '{model_path}'.\n\n"
                f"Free disk space or change the cache directory:\n"
                f"  export HF_HOME=/path/to/large/disk"
            )
        raise
