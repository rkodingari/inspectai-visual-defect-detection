from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a model comparison table from evaluation JSON files")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_comparison.md"))
    args = parser.parse_args()
    metrics = [json.loads(path.read_text()) for path in args.reports]
    header = "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE | Latency (ms) | Size (MB) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [
        f"| {m['model']} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
        f"{m['f1']:.3f} | {m['roc_auc']:.3f} | {m['expected_calibration_error']:.3f} | "
        f"{m['inference_latency_ms_per_image']:.1f} | {m['model_size_mb']:.1f} |"
        for m in metrics
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join([header, *rows]) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
