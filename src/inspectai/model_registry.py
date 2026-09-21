from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspectai.config import ARTIFACT_DIR, DEFAULT_CHECKPOINT
from inspectai.models.factory import MODEL_DISPLAY_NAMES

DEFAULT_REGISTRY = ARTIFACT_DIR / "model_registry.json"


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {"best_model": "best", "models": {}}


def available_models(path: str | Path = DEFAULT_REGISTRY) -> dict[str, dict[str, Any]]:
    path = Path(path)
    registry = load_registry(path)
    registered: dict[str, dict[str, Any]] = {}
    for name, metadata in registry.get("models", {}).items():
        checkpoint = Path(metadata["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        if checkpoint.exists():
            registered[name] = {
                **metadata,
                "checkpoint": str(checkpoint),
                "display_name": metadata.get("display_name", MODEL_DISPLAY_NAMES.get(name, name)),
            }
    available: dict[str, dict[str, Any]] = {}
    best_name = registry.get("best_model")
    if best_name in registered:
        available["best"] = {
            **registered[best_name],
            "display_name": f"Best available — {registered[best_name]['display_name']}",
        }
    elif DEFAULT_CHECKPOINT.exists():
        available["best"] = {
            "display_name": "Best available (recommended)",
            "checkpoint": str(DEFAULT_CHECKPOINT),
        }
    available.update(registered)
    return available


def resolve_checkpoint(model_name: str = "best", path: str | Path = DEFAULT_REGISTRY) -> Path:
    models = available_models(path)
    if model_name not in models:
        choices = ", ".join(models) or "none"
        raise ValueError(f"Model '{model_name}' is unavailable. Available models: {choices}")
    return Path(models[model_name]["checkpoint"])
