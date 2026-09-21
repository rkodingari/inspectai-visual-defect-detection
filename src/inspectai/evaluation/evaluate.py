from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support, roc_auc_score,
                             RocCurveDisplay)
from torch.utils.data import DataLoader

from inspectai.config import CLASS_NAMES, select_device
from inspectai.data.dataset import BinaryDefectDataset, build_transforms
from inspectai.models.factory import create_model


def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    targets, logits, paths = [], [], []
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for images, labels, batch_paths in loader:
            outputs = model(images.to(device))
            logits.append(outputs.cpu().numpy())
            targets.extend(labels.numpy())
            paths.extend(batch_paths)
    latency_ms = (time.perf_counter() - started) * 1000 / len(loader.dataset)
    return np.asarray(targets), np.concatenate(logits), paths, latency_ms


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_tensor / log_temperature.exp(), targets_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 10.0))


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def evaluate(checkpoint_path: Path, data_dir: Path, output_dir: Path,
             batch_size: int = 16, device_name: str = "auto") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    image_size = checkpoint.get("image_size", 224)
    val = BinaryDefectDataset(data_dir, "val", build_transforms(False, image_size))
    test = BinaryDefectDataset(data_dir, "test", build_transforms(False, image_size))
    val_targets, val_logits, _, _ = collect_predictions(model, DataLoader(val, batch_size=batch_size), device)
    temperature = fit_temperature(val_logits, val_targets)
    targets, logits, paths, latency = collect_predictions(model, DataLoader(test, batch_size=batch_size), device)
    probabilities = torch.softmax(torch.tensor(logits) / temperature, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average="binary", zero_division=0)
    report = classification_report(targets, predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metrics = {
        "model": checkpoint["model_name"], "model_version": checkpoint.get("version", "unknown"),
        "accuracy": accuracy_score(targets, predictions), "precision": precision,
        "recall": recall, "f1": f1,
        "roc_auc": roc_auc_score(targets, probabilities[:, 1]) if len(set(targets)) == 2 else None,
        "temperature": temperature,
        "expected_calibration_error": expected_calibration_error(probabilities, targets),
        "inference_latency_ms_per_image": latency,
        "model_size_mb": checkpoint_path.stat().st_size / (1024 ** 2),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "false_positives": [path for path, y, pred in zip(paths, targets, predictions) if y == 0 and pred == 1][:8],
        "false_negatives": [path for path, y, pred in zip(paths, targets, predictions) if y == 1 and pred == 0][:8],
        "evidence_note": "Metrics apply only to this prepared split and are not production claims.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    ConfusionMatrixDisplay.from_predictions(targets, predictions, display_labels=CLASS_NAMES, cmap="Blues")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    if len(set(targets)) == 2:
        RocCurveDisplay.from_predictions(targets, probabilities[:, 1])
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=160)
        plt.close()
    prob_true, prob_pred = calibration_curve(targets, probabilities[:, 1], n_bins=8)
    plt.plot([0, 1], [0, 1], "--", label="ideal")
    plt.plot(prob_pred, prob_true, marker="o", label="model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction defective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=160)
    plt.close()
    error_indices = np.flatnonzero(targets != predictions)[:8]
    if len(error_indices):
        columns = min(4, len(error_indices))
        rows = int(np.ceil(len(error_indices) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, index in zip(axes.flat, error_indices):
            axis.imshow(plt.imread(paths[index]))
            error_type = "FP" if targets[index] == 0 else "FN"
            axis.set_title(f"{error_type}: p(defect)={probabilities[index, 1]:.2f}")
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / "misclassification_examples.png", dpi=160)
        plt.close(figure)
    report_lines = [
        f"# Evaluation report — {metrics['model']}", "", metrics["evidence_note"], "",
        "| Metric | Value |", "|---|---:|",
        *[
            f"| {label} | {metrics[key]:.4f} |"
            for label, key in (("Accuracy", "accuracy"), ("Precision", "precision"),
                               ("Recall", "recall"), ("F1", "f1"),
                               ("ROC-AUC", "roc_auc"),
                               ("Expected calibration error", "expected_calibration_error"),
                               ("Latency (ms/image)", "inference_latency_ms_per_image"),
                               ("Model size (MB)", "model_size_mb"))
        ],
        "", "## Per-class results", "",
        "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|",
        *[
            f"| {name} | {report[name]['precision']:.4f} | {report[name]['recall']:.4f} | "
            f"{report[name]['f1-score']:.4f} | {int(report[name]['support'])} |"
            for name in CLASS_NAMES
        ],
        "", "## Error inspection", "",
        f"False positives: {len(metrics['false_positives'])} paths recorded.", "",
        f"False negatives: {len(metrics['false_negatives'])} paths recorded.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, checkpoint_path)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and calibrate InspectAI")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.output_dir, args.batch_size, args.device)


if __name__ == "__main__":
    main()
