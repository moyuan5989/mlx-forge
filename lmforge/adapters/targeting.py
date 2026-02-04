"""Glob-based adapter targeting for LMForge v0.

Module path matching uses fnmatch.fnmatch() on dot-separated paths.
See V0_DESIGN_FREEZE.md §4 for full semantics.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Optional

PRESETS: dict[str, list[str]] = {
    "attention-qv": ["*.self_attn.q_proj", "*.self_attn.v_proj"],
    "attention-all": [
        "*.self_attn.q_proj",
        "*.self_attn.k_proj",
        "*.self_attn.v_proj",
        "*.self_attn.o_proj",
    ],
    "mlp": ["*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"],
    "all-linear": [
        "*.self_attn.q_proj",
        "*.self_attn.k_proj",
        "*.self_attn.v_proj",
        "*.self_attn.o_proj",
        "*.mlp.gate_proj",
        "*.mlp.up_proj",
        "*.mlp.down_proj",
    ],
}


def get_patterns(config) -> list[str]:
    """Resolve adapter config to a list of glob patterns.

    Uses config.targets if provided, otherwise resolves config.preset via PRESETS.
    """
    if config.targets is not None:
        return config.targets

    if config.preset is not None:
        if config.preset not in PRESETS:
            raise ValueError(
                f"Unknown preset '{config.preset}'. "
                f"Available: {list(PRESETS.keys())}"
            )
        return PRESETS[config.preset]

    # This should never happen due to config validation, but be defensive
    raise ValueError("No adapter targets specified (neither 'targets' nor 'preset').")


def named_modules(module, prefix: str = ""):
    """Yield (name, module) pairs for all submodules recursively.

    For MLX nn.Module, we use the children() method which returns a dict
    of child modules. Lists are traversed with numeric indices.

    Produces paths like:
        - model.layers.0.self_attn.q_proj
        - model.layers.0.mlp.gate_proj
        - model.embed_tokens
    """
    yield prefix, module

    # MLX modules have a children() method that returns {name: child_module_or_list}
    if hasattr(module, "children"):
        children = module.children()
        if isinstance(children, dict):
            for name, child in children.items():
                full_name = f"{prefix}.{name}" if prefix else name
                # Handle lists of modules (e.g., transformer layers)
                if isinstance(child, list):
                    for idx, item in enumerate(child):
                        item_name = f"{full_name}.{idx}"
                        yield from named_modules(item, item_name)
                else:
                    yield from named_modules(child, full_name)


def resolve_targets(
    model, patterns: list[str], num_layers: Optional[int] = None
) -> list[tuple[str, object]]:
    """Match glob patterns against model module paths.

    Returns list of (path, module) tuples for matched modules.
    Raises ValueError if no modules match, listing the attempted patterns
    and first 20 available module paths.
    """
    # Get all module paths
    all_modules = list(named_modules(model))

    # If num_layers is specified, determine total layer count and filter
    total_layers = None
    if num_layers is not None:
        total_layers = _count_transformer_layers(all_modules)
        if total_layers is None:
            raise ValueError(
                f"Could not determine total layer count for num_layers={num_layers} filtering. "
                "The model may not have a standard transformer structure with 'layers.N' paths."
            )

    # Match patterns
    matched = []
    for name, module in all_modules:
        if not name:  # Skip root module
            continue

        # Apply num_layers filter
        if num_layers is not None:
            layer_idx = _extract_layer_index(name)
            if layer_idx is None:
                # Skip modules without layer index when num_layers is set
                continue
            if layer_idx < total_layers - num_layers:
                # Skip layers outside the last N
                continue

        # Match against patterns
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                matched.append((name, module))
                break  # Don't match the same module multiple times

    # Error if no matches
    if not matched:
        available_paths = [name for name, _ in all_modules if name][:20]
        raise ValueError(
            f"No modules matched patterns {patterns}.\n"
            f"Available paths (first 20): {available_paths}"
        )

    return matched


def _count_transformer_layers(all_modules: list[tuple[str, object]]) -> Optional[int]:
    """Count transformer layers by finding the maximum layer index.

    Returns None if no layer indices found.
    """
    max_layer_idx = -1
    for name, _ in all_modules:
        idx = _extract_layer_index(name)
        if idx is not None:
            max_layer_idx = max(max_layer_idx, idx)

    if max_layer_idx >= 0:
        return max_layer_idx + 1  # Convert from 0-indexed to count
    return None


def _extract_layer_index(path: str) -> Optional[int]:
    """Extract layer index from a module path like 'model.layers.15.self_attn.q_proj'.

    Returns None if no layer index found.
    """
    # Match patterns like "layers.N" or "layer.N"
    match = re.search(r'\.layers?\.(\d+)\.', path)
    if match:
        return int(match.group(1))
    return None
