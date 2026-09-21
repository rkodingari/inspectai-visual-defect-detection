from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PredictionStore:
    def __init__(self, path: str | Path = "logs/inspectai.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT,
                    label TEXT NOT NULL, confidence REAL NOT NULL, latency_ms REAL NOT NULL,
                    model_version TEXT NOT NULL, probabilities TEXT NOT NULL, feedback TEXT
                )
            """)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def add(self, filename: str, prediction: Any) -> str:
        prediction_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (prediction_id, datetime.now(timezone.utc).isoformat(), filename,
                 prediction.label, prediction.confidence, prediction.latency_ms,
                 prediction.model_version, json.dumps(prediction.probabilities)),
            )
        return prediction_id

    def feedback(self, prediction_id: str, feedback: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE predictions SET feedback = ? WHERE id = ?", (feedback, prediction_id)
            )
        return cursor.rowcount == 1

    def recent(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

