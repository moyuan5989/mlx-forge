"""Centralized input validation for Studio API.

Prevents path traversal, null-byte injection, and similar attacks.
"""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


def validate_safe_name(name: str, label: str = "name") -> str:
    """Validate that name contains no path separators or traversal.

    Allows alphanumeric, dots, hyphens, underscores. Must start with alphanumeric.
    Max 128 chars.

    Raises:
        ValueError: If name is invalid.
    """
    if not isinstance(name, str) or not _SAFE_NAME.match(name):
        raise ValueError(f"Invalid {label}: {name!r}")
    return name


def validate_safe_path(requested: Path, root: Path) -> Path:
    """Ensure resolved path stays within root directory.

    Raises:
        ValueError: If path escapes root or contains null bytes.
    """
    req_str = str(requested)
    if "\x00" in req_str:
        raise ValueError(f"Path contains null byte: {requested}")
    resolved = (root / requested).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {requested}")
    return resolved
