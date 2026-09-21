from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from inspectai.config import ARTIFACT_DIR
from inspectai.models.factory import MODEL_DISPLAY_NAMES, SUPERVISED_MODELS
from inspectai.training.train import TrainConfig, train


def train_all(data_dir: str, output_dir: Path = ARTIFACT_DIR, epochs: int = 8,
              batch_size: int = 16, device: str = "auto", quick: bool = False,
              pretrained: bool = True) -> Path:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    registry: dict = {"selection_metric": "validation_f1", "models": {}}
    for model_name in SUPERVISED_MODELS:
        run_output = output_dir / "model_runs" / model_name
        checkpoint = train(TrainConfig(
            data_dir=data_dir, model=model_name, epochs=epochs, batch_size=batch_size,
            device=device, quick=quick, pretrained=pretrained, output_dir=str(run_output),
        ))
        target = model_dir / f"{model_name}.pt"
        shutil.copy2(checkpoint, target)
        saved = torch.load(target, map_location="cpu", weights_only=False)
        registry["models"][model_name] = {
            "display_name": MODEL_DISPLAY_NAMES[model_name],
            "checkpoint": str(target.relative_to(output_dir)),
            "best_val_f1": float(saved["best_val_f1"]),
            "model_version": saved["version"],
        }
    best_name = max(registry["models"], key=lambda name: registry["models"][name]["best_val_f1"])
    registry["best_model"] = best_name
    shutil.copy2(registry["models"][best_name]["checkpoint"], output_dir / "best_model.pt")
    registry_path = output_dir / "model_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2))
    print(json.dumps(registry, indent=2))
    return registry_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register all supervised InspectAI models")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    train_all(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.device,
              args.quick, not args.no_pretrained)


if __name__ == "__main__":
    main()
