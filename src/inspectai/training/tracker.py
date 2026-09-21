from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class JsonCsvTracker:
    """Dependency-free local experiment tracker."""

    def __init__(self, run_dir: Path, config: dict[str, Any]):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        self.metrics_path = self.run_dir / "metrics.csv"

    def log_epoch(self, metrics: dict[str, Any]) -> None:
        exists = self.metrics_path.exists()
        with self.metrics_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            if not exists:
                writer.writeheader()
            writer.writerow(metrics)

    def save_summary(self, summary: dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

