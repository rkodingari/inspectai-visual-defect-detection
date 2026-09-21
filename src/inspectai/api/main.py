from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from inspectai import __version__
from inspectai.inference import Predictor, image_from_bytes
from inspectai.model_registry import available_models, resolve_checkpoint
from inspectai.monitoring import monitoring_summary
from inspectai.storage import PredictionStore

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
app = FastAPI(title="InspectAI API", version=__version__)


@lru_cache(maxsize=8)
def get_predictor(model_name: str = "best") -> Predictor:
    explicit_path = os.getenv("INSPECTAI_MODEL_PATH")
    checkpoint = Path(explicit_path) if explicit_path else resolve_checkpoint(model_name)
    return Predictor(checkpoint)


@lru_cache(maxsize=1)
def get_store() -> PredictionStore:
    return PredictionStore(os.getenv("INSPECTAI_DB_PATH", "logs/inspectai.db"))


class FeedbackRequest(BaseModel):
    prediction_id: str
    feedback: str = Field(pattern="^(correct|incorrect|unsure)$")


@app.get("/health")
def health() -> dict:
    models = available_models()
    return {"status": "ok", "model_available": bool(models),
            "available_models": list(models), "version": __version__}


@app.get("/models")
def models() -> dict:
    return {"models": available_models()}


@app.post("/predict")
async def predict(file: UploadFile = File(...), explain: bool = True, model: str = "best") -> dict:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are accepted")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB")
    try:
        image = image_from_bytes(content)
        result = get_predictor(model).predict(image, explain=explain)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400 if isinstance(error, ValueError) else 503, detail=str(error)) from error
    prediction_id = get_store().add(file.filename or "upload", result)
    encoded_heatmap = None
    if result.heatmap is not None:
        buffer = io.BytesIO()
        result.heatmap.save(buffer, format="JPEG", quality=88)
        encoded_heatmap = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "prediction_id": prediction_id, "label": result.label,
        "confidence": result.confidence, "probabilities": result.probabilities,
        "latency_ms": result.latency_ms, "model_version": result.model_version,
        "requested_model": model,
        "gradcam_base64": encoded_heatmap,
    }


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict:
    if not get_store().feedback(request.prediction_id, request.feedback):
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"stored": True}


@app.get("/monitoring")
def monitoring(limit: int = 200) -> dict:
    return monitoring_summary(get_store(), min(max(limit, 1), 1000))
