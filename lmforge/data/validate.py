"""Data quality validation for LMForge datasets.

Checks:
- Role alternation (user/assistant)
- No trailing user turns
- No empty messages
- Token length distribution stats
- Duplicate detection
- Train/val overlap detection
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lmforge.data.formats import detect_format, validate_samples


@dataclass
class ValidationReport:
    """Results of data validation."""

    num_samples: int = 0
    format: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    length_stats: dict = field(default_factory=dict)
    num_duplicates: int = 0
    overlap_count: int = 0

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_file(
    path: str,
    *,
    val_path: Optional[str] = None,
) -> ValidationReport:
    """Validate a JSONL data file.

    Args:
        path: Path to JSONL file to validate.
        val_path: Optional path to validation set for overlap detection.

    Returns:
        ValidationReport with errors, warnings, and statistics.
    """
    report = ValidationReport()

    # Load samples
    data_path = Path(path)
    if not data_path.exists():
        report.errors.append(f"File not found: {path}")
        return report

    with open(data_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    report.num_samples = len(samples)
    if not samples:
        report.errors.append("File is empty (no samples)")
        return report

    # Detect format
    fmt = detect_format(samples)
    report.format = fmt

    # Run basic schema validation
    schema_errors = validate_samples(samples, fmt)
    report.errors.extend(schema_errors)

    # Format-specific quality checks
    if fmt == "chat":
        _validate_chat_quality(samples, report)
    elif fmt == "preference":
        _validate_preference_quality(samples, report)

    # Length stats (character-level)
    _compute_length_stats(samples, fmt, report)

    # Duplicate detection
    _detect_duplicates(samples, report)

    # Train/val overlap
    if val_path:
        _detect_overlap(samples, val_path, fmt, report)

    return report


def _validate_chat_quality(samples: list[dict], report: ValidationReport) -> None:
    """Check chat-specific quality issues."""
    for idx, sample in enumerate(samples):
        messages = sample.get("messages", [])
        if not isinstance(messages, list):
            continue

        # Check for empty messages
        for msg_idx, msg in enumerate(messages):
            if isinstance(msg, dict) and not msg.get("content", "").strip():
                report.warnings.append(
                    f"Sample {idx}, message {msg_idx}: empty content"
                )

        # Check role alternation
        roles = [m.get("role") for m in messages if isinstance(m, dict)]
        for i in range(1, len(roles)):
            if roles[i] == roles[i - 1] and roles[i] in ("user", "assistant"):
                report.warnings.append(
                    f"Sample {idx}: consecutive '{roles[i]}' roles at positions {i-1},{i}"
                )

        # Check for trailing user turn (no assistant response)
        if roles and roles[-1] == "user":
            report.warnings.append(
                f"Sample {idx}: ends with user turn (no assistant response)"
            )


def _validate_preference_quality(
    samples: list[dict], report: ValidationReport
) -> None:
    """Check preference-specific quality issues."""
    for idx, sample in enumerate(samples):
        for field_name in ("chosen", "rejected"):
            messages = sample.get(field_name, [])
            if not isinstance(messages, list):
                continue

            # Empty content check
            for msg_idx, msg in enumerate(messages):
                if isinstance(msg, dict) and not msg.get("content", "").strip():
                    report.warnings.append(
                        f"Sample {idx}, {field_name}[{msg_idx}]: empty content"
                    )

            # Trailing user turn
            roles = [m.get("role") for m in messages if isinstance(m, dict)]
            if roles and roles[-1] == "user":
                report.warnings.append(
                    f"Sample {idx}: {field_name} ends with user turn"
                )


def _compute_length_stats(
    samples: list[dict], fmt: str, report: ValidationReport
) -> None:
    """Compute token-approximate length stats (character-based)."""
    lengths = []
    for sample in samples:
        if fmt == "chat":
            msgs = sample.get("messages", [])
            length = sum(len(m.get("content", "")) for m in msgs if isinstance(m, dict))
        elif fmt == "completions":
            length = len(sample.get("prompt", "")) + len(sample.get("completion", ""))
        elif fmt == "text":
            length = len(sample.get("text", ""))
        elif fmt == "preference":
            chosen = sample.get("chosen", [])
            rejected = sample.get("rejected", [])
            c_len = sum(len(m.get("content", "")) for m in chosen if isinstance(m, dict))
            r_len = sum(len(m.get("content", "")) for m in rejected if isinstance(m, dict))
            length = max(c_len, r_len)
        else:
            length = 0
        lengths.append(length)

    if lengths:
        lengths.sort()
        n = len(lengths)
        report.length_stats = {
            "min": lengths[0],
            "max": lengths[-1],
            "mean": round(sum(lengths) / n),
            "p50": lengths[n // 2],
            "p95": lengths[int(n * 0.95)],
        }


def _sample_fingerprint(sample: dict) -> str:
    """Create a content hash for duplicate detection."""
    return hashlib.md5(
        json.dumps(sample, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _detect_duplicates(samples: list[dict], report: ValidationReport) -> None:
    """Count exact duplicate samples."""
    seen = Counter(_sample_fingerprint(s) for s in samples)
    report.num_duplicates = sum(c - 1 for c in seen.values() if c > 1)
    if report.num_duplicates > 0:
        report.warnings.append(
            f"{report.num_duplicates} duplicate sample(s) found"
        )


def _detect_overlap(
    train_samples: list[dict],
    val_path: str,
    fmt: str,
    report: ValidationReport,
) -> None:
    """Detect overlap between train and validation sets."""
    val_file = Path(val_path)
    if not val_file.exists():
        report.warnings.append(f"Validation file not found: {val_path}")
        return

    with open(val_file) as f:
        val_samples = [json.loads(line) for line in f if line.strip()]

    train_fps = {_sample_fingerprint(s) for s in train_samples}
    val_fps = {_sample_fingerprint(s) for s in val_samples}

    overlap = train_fps & val_fps
    report.overlap_count = len(overlap)
    if overlap:
        report.warnings.append(
            f"{len(overlap)} sample(s) appear in both train and validation sets"
        )
