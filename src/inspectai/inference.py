from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inspectai.config import DEFAULT_CHECKPOINT, select_device
from inspectai.data.dataset import build_transforms
from inspectai.explainability.gradcam import GradCAM, overlay_heatmap
from inspectai.models.factory import create_model


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_version: str
    heatmap: Image.Image | None = None


class Predictor:
    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}. Run the training command first."
            )
        self.device = select_device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model_name = checkpoint["model_name"]
        self.model = create_model(self.model_name, pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.class_names = checkpoint.get("class_names", ["normal", "defective"])
        self.image_size = int(checkpoint.get("image_size", 224))
        self.version = checkpoint.get("version", self.checkpoint_path.stem)
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.transform = build_transforms(False, self.image_size)

    def predict(self, image: Image.Image, explain: bool = True) -> Prediction:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor) / self.temperature
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        index = int(np.argmax(probabilities))
        heatmap_image = None
        if explain:
            cam = GradCAM(self.model, self.model.gradcam_layer)
            try:
                heatmap_image = overlay_heatmap(image, cam(tensor, index))
            finally:
                cam.close()
        latency_ms = (time.perf_counter() - started) * 1000
        return Prediction(
            label=self.class_names[index], confidence=float(probabilities[index]),
            probabilities={name: float(probabilities[i]) for i, name in enumerate(self.class_names)},
            latency_ms=latency_ms, model_version=self.version, heatmap=heatmap_image,
        )


def image_from_bytes(content: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as error:
        raise ValueError("Uploaded file is not a valid image") from error

