from __future__ import annotations

import json
from collections import Counter
from typing import Any

from inspectai.storage import PredictionStore


def monitoring_summary(store: PredictionStore, limit: int = 200) -> dict[str, Any]:
    """Aggregate lightweight operational metrics from recent local predictions."""
    rows = store.recent(limit)
    if not rows:
        return {"prediction_count": 0, "feedback_count": 0}
    labels = Counter(row["label"] for row in rows)
    feedback_rows = [row for row in rows if row["feedback"]]
    correct = sum(row["feedback"] == "correct" for row in feedback_rows)
    probabilities = [json.loads(row["probabilities"])["defective"] for row in rows]
    return {
        "prediction_count": len(rows),
        "feedback_count": len(feedback_rows),
        "feedback_accuracy": correct / len(feedback_rows) if feedback_rows else None,
        "defective_prediction_rate": labels["defective"] / len(rows),
        "mean_defective_probability": sum(probabilities) / len(probabilities),
        "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
        "window_size": limit,
        "warning": "Feedback metrics are observational and may be biased; investigate shifts rather than treating them as ground truth.",
    }
