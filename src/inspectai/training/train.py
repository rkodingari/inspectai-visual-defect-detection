from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from inspectai.config import ARTIFACT_DIR, CLASS_NAMES, seed_everything, select_device
from inspectai.data.dataset import BinaryDefectDataset, build_transforms, validate_dataset
from inspectai.models.factory import SUPERVISED_MODELS, create_model, trainable_parameters
from inspectai.training.tracker import JsonCsvTracker


@dataclass
class TrainConfig:
    data_dir: str
    model: str = "mobilenet_v3_small"
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 224
    seed: int = 42
    device: str = "auto"
    pretrained: bool = True
    freeze_backbone: bool = True
    quick: bool = False
    num_workers: int = 0
    output_dir: str = str(ARTIFACT_DIR)


def make_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_set = BinaryDefectDataset(config.data_dir, "train", build_transforms(True, config.image_size))
    val_set = BinaryDefectDataset(config.data_dir, "val", build_transforms(False, config.image_size))
    labels = np.array([label for _, label in train_set.samples])
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / np.maximum(counts, 1)
    sampler = WeightedRandomSampler(weights[labels].tolist(), len(labels), replacement=True)
    kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=False)
    return (
        DataLoader(train_set, sampler=sampler, **kwargs),
        DataLoader(val_set, shuffle=False, **kwargs),
    )


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, predictions, targets = 0.0, [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
        targets.extend(labels.cpu().tolist())
    accuracy = float(np.mean(np.array(predictions) == np.array(targets)))
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy,
        "f1": f1_score(targets, predictions, average="binary", zero_division=0),
    }


def train(config: TrainConfig) -> Path:
    seed_everything(config.seed)
    counts = validate_dataset(config.data_dir)
    if config.quick:
        config.epochs = min(config.epochs, 2)
        config.image_size = min(config.image_size, 128)
    device = select_device(config.device)
    train_loader, val_loader = make_loaders(config)
    model = create_model(config.model, config.pretrained, config.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config.output_dir) / "runs" / f"{stamp}_{config.model}"
    tracker = JsonCsvTracker(run_dir, {**asdict(config), "device_used": str(device), "counts": counts})
    checkpoint_path = Path(config.output_dir) / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        tracker.log_epoch(row)
        print(json.dumps(row))
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            trainable, total = trainable_parameters(model)
            torch.save({
                "state_dict": model.state_dict(), "model_name": config.model,
                "class_names": CLASS_NAMES, "image_size": config.image_size,
                "version": f"{config.model}-{stamp}", "best_val_f1": best_f1,
                "config": asdict(config), "trainable_parameters": trainable,
                "total_parameters": total,
            }, checkpoint_path)
    summary = {"best_val_f1": best_f1, "elapsed_seconds": time.perf_counter() - started,
               "checkpoint": str(checkpoint_path)}
    tracker.save_summary(summary)
    (Path(config.output_dir) / "latest_run.json").write_text(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    print(json.dumps(summary, indent=2))
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train InspectAI classifier")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", choices=SUPERVISED_MODELS, default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_dir=args.data_dir, model=args.model, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        image_size=args.image_size, seed=args.seed, device=args.device,
        output_dir=args.output_dir, num_workers=args.num_workers,
        quick=args.quick, pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze,
    )
    train(config)


if __name__ == "__main__":
    main()
