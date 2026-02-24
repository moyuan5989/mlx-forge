"""Real-time metrics.jsonl file watcher.

Uses seek-based polling to efficiently read new lines
appended to a metrics file.
"""

from __future__ import annotations

import json
from pathlib import Path


class MetricsWatcher:
    """Watches a metrics.jsonl file for new entries via polling."""

    def __init__(self, metrics_path: str | Path):
        self.metrics_path = Path(metrics_path)
        self._offset = 0

        # Start at end of file if it exists
        if self.metrics_path.exists():
            self._offset = self.metrics_path.stat().st_size

    def poll(self) -> list[dict]:
        """Read new lines since last poll.

        Returns list of parsed JSON dicts (may be empty).
        """
        if not self.metrics_path.exists():
            return []

        current_size = self.metrics_path.stat().st_size
        if current_size <= self._offset:
            # File hasn't grown (or was truncated)
            if current_size < self._offset:
                self._offset = 0  # Reset on truncation
            return []

        new_entries = []
        with open(self.metrics_path) as f:
            f.seek(self._offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    new_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self._offset = f.tell()

        return new_entries

    def reset(self):
        """Reset offset to beginning of file."""
        self._offset = 0
