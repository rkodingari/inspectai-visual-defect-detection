# InspectAI — Visual Defect Detection

InspectAI is a laptop-friendly, end-to-end computer-vision portfolio project. It classifies product images as **normal** or **defective**, reports confidence and latency, explains predictions with Grad-CAM, exposes FastAPI and Streamlit applications, and records predictions plus human feedback for lightweight monitoring. A trained checkpoint remains specific to the product domain represented by its training data.

The primary real dataset is the **bottle** category of [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). A deterministic synthetic dataset is included through a generator, so every pipeline and test can run without a network connection. This is an educational supervised-classification adaptation—not the canonical unsupervised MVTec benchmark and not a production quality-control claim. MVTec AD is licensed CC BY-NC-SA 4.0; review its non-commercial terms before use.

## What this demonstrates

- Reproducible data validation, deterministic splitting, augmentation, and imbalance-aware sampling
- A small CNN baseline plus MobileNetV3-Small, EfficientNet-B0, and ResNet18 transfer learning
- Accuracy, precision, recall, F1, per-class metrics, ROC-AUC, confusion matrix, calibration/ECE, latency, and model size
- Temperature scaling fitted only on validation data
- Grad-CAM explanations and selectable trained checkpoints
- FastAPI inference, Streamlit UI, SQLite prediction/feedback storage, and rolling monitoring metrics
- Local JSON/CSV experiment tracking, tests, Docker, and GitHub Actions

## Repository structure

```text
.
├── data/                         # raw/ and processed/ are generated
├── notebooks/01_eda.ipynb       # counts, dimensions, balance, image samples
├── scripts/compare_models.py    # builds Markdown comparison table
├── src/inspectai/
│   ├── api/main.py              # FastAPI endpoints
│   ├── data/                    # preparation, validation, dataset, transforms
│   ├── evaluation/              # metrics, plots, calibration, error examples
│   ├── explainability/          # framework-light Grad-CAM
│   ├── models/                  # four supervised classifier architectures
│   ├── training/                # single/all-model training and JSON/CSV tracking
│   ├── ui/app.py                # Streamlit application
│   ├── inference.py             # shared inference contract
│   ├── monitoring.py            # rolling operational/feedback metrics
│   └── storage.py               # local SQLite store
├── tests/
├── Dockerfile
└── pyproject.toml
```

## Quick start (offline-capable)

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'

# Generate 80 tiny images and train a 2-epoch CPU baseline
make quick

# Evaluate, calibrate, and create plots
python -m inspectai.evaluation.evaluate \
  --checkpoint artifacts/best_model.pt \
  --data-dir data/processed/synthetic \
  --output-dir artifacts/evaluation/baseline_synthetic

pytest
```

Synthetic images are deliberately easy and exist only to verify plumbing. Their metrics say nothing about real manufacturing performance.

## Real dataset setup

The script first tries the category-specific public archive, which is much smaller than the complete MVTec archive:

```bash
python -m inspectai.data.prepare \
  --dataset mvtec --category bottle \
  --output data/processed/mvtec_bottle
```

MVTec's current site may require its download form. If automatic download is unavailable, download `bottle.tar.xz` from the official MVTec AD page, extract it, then run:

```bash
python -m inspectai.data.prepare \
  --dataset mvtec --category bottle \
  --source /path/to/extracted/root \
  --output data/processed/mvtec_bottle
```

Add `--quick` to cap each source group for a faster experiment and `--force` to recreate an existing prepared directory. The script maps `train/good` and `test/good` to normal; known test anomaly types are split deterministically into supervised train/validation/test groups. This deliberate protocol change is saved in `dataset_metadata.json` and prevents comparison with published MVTec anomaly-detection scores.

Explore the prepared data with:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Training

Baseline:

```bash
python -m inspectai.training.train \
  --data-dir data/processed/mvtec_bottle \
  --model baseline_cnn --epochs 12 --batch-size 16
```

Transfer learning (the default downloads ImageNet weights once, freezes most feature layers, and fine-tunes the final feature block and classifier):

```bash
python -m inspectai.training.train \
  --data-dir data/processed/mvtec_bottle \
  --model mobilenet_v3_small --epochs 8 --batch-size 16
```

Available model names are `baseline_cnn`, `mobilenet_v3_small`, `efficientnet_b0`, and `resnet18`. To train and register all four statically:

```bash
python -m inspectai.training.train_all \
  --data-dir data/processed/mvtec_bottle \
  --epochs 8 --batch-size 16
```

This creates `artifacts/models/<model>.pt` and `artifacts/model_registry.json`. The model with the highest validation F1 becomes “Best available” and is also copied to `artifacts/best_model.pt`. The test split is not used for model selection.

Use `--device cpu`, `--device cuda`, or `--device mps` to override auto-selection. Use `--quick`, `--no-pretrained`, or `--unfreeze` for short smoke runs, offline architecture tests, or full fine-tuning. Training uses a weighted sampler to reduce class-imbalance effects. Seeds cover Python, NumPy, and PyTorch; exact floating-point reproducibility can still vary across devices and library versions.

Each run writes config, epoch metrics, and a summary under `artifacts/`. Checkpoints contain model identity, image size, class names, parameter counts, version, and training configuration.

## Evaluation and model comparison

After static multi-model training, evaluate any registered checkpoint with the same test split:

```bash
python -m inspectai.evaluation.evaluate \
  --checkpoint artifacts/models/baseline_cnn.pt --data-dir data/processed/mvtec_bottle \
  --output-dir artifacts/evaluation/baseline

python -m inspectai.evaluation.evaluate \
  --checkpoint artifacts/models/efficientnet_b0.pt --data-dir data/processed/mvtec_bottle \
  --output-dir artifacts/evaluation/efficientnet

python scripts/compare_models.py \
  artifacts/evaluation/baseline/metrics.json \
  artifacts/evaluation/efficientnet/metrics.json
```

The evaluator produces `metrics.json`, `report.md`, `confusion_matrix.png`, `roc_curve.png`, `calibration_curve.png`, and a grid of `misclassification_examples.png` when errors exist. It records up to eight false-positive and false-negative paths for inspection and saves the validation-fitted temperature into the checkpoint.

### Verified synthetic smoke results

These numbers come from the deterministic two-epoch quick run on 2026-06-29 using an Apple Silicon CPU. They verify code execution only; CI and the container target Python 3.11, while latency varies by hardware and excludes Grad-CAM rendering.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE | Latency (ms/image) | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 0.500 | 0.500 | 1.000 | 0.667 | 1.000 | 0.029 | 2.6 | 0.1 |
| MobileNetV3-Small (ImageNet transfer) | 0.625 | 1.000 | 0.250 | 0.400 | 1.000 | 0.273 | 11.5 | 5.9 |
| EfficientNet-B0 (ImageNet transfer) | 0.938 | 1.000 | 0.875 | 0.933 | 1.000 | 0.230 | 33.1 | 15.6 |
| ResNet18 (ImageNet transfer) | 0.812 | 1.000 | 0.625 | 0.769 | 1.000 | 0.214 | 14.6 | 42.7 |

Do not interpret synthetic accuracy as evidence of defect-detection quality. For a credible portfolio result, run all models on the prepared real split, retain the generated artifacts, and report the data version and hardware alongside this table.

## Applications

Start the API after training:

```bash
uvicorn inspectai.api.main:app --reload
curl http://127.0.0.1:8000/models
curl -F "file=@sample.png" "http://127.0.0.1:8000/predict?model=efficientnet_b0&explain=true"
curl http://127.0.0.1:8000/monitoring
```

Endpoints:

- `GET /health`: service and checkpoint availability
- `GET /models`: registered checkpoints available for inference
- `POST /predict`: multipart image plus optional `model`; prediction, confidence, probabilities, latency, version, and base64 Grad-CAM
- `POST /feedback`: `{"prediction_id":"...","feedback":"correct|incorrect|unsure"}`
- `GET /monitoring`: rolling prediction rate, confidence, latency, and optional feedback accuracy

Run the UI in another terminal:

```bash
streamlit run src/inspectai/ui/app.py
```

The model dropdown lists only checkpoints that exist. “Best available” uses the validation-F1 winner recorded by static multi-model training.

Predictions and optional feedback are stored in `logs/inspectai.db`. Monitoring is intentionally lightweight: it helps reveal changes in output mix, latency, and user feedback, but it is not automatic drift proof. Reliable drift detection requires representative labeled production samples and an alert policy.

## Docker

Train or place a compatible checkpoint at `artifacts/best_model.pt` before building:

```bash
docker build -t inspectai .
docker run --rm -p 8000:8000 \
  -v "$PWD/logs:/app/logs" inspectai
```

The image serves FastAPI on port 8000. Model training is intentionally kept outside the runtime image.

## Tests and verified commands

```bash
pytest                         # data, models, Grad-CAM, storage, API, training smoke
ruff check src tests
python -m inspectai.data.prepare --dataset synthetic --output data/processed/synthetic --force
python -m inspectai.training.train --data-dir data/processed/synthetic --model baseline_cnn --epochs 2 --quick
python -m inspectai.training.train_all --data-dir data/processed/synthetic --epochs 2 --quick
```

GitHub Actions runs Ruff and Pytest on Python 3.11. The training smoke test uses 64×64 synthetic images and never downloads pretrained weights.

## Limitations and responsible interpretation

- Binary classification answers “does this resemble a known defect?” It does not localize, measure, or name defects; Grad-CAM is a coarse explanation, not a segmentation mask.
- MVTec has no defective training images in its canonical protocol. Reusing known anomalies for supervised training changes the scientific question and can overstate generalization to unseen defect types.
- The small dataset can produce high-variance metrics. Use repeated splits or cross-validation before drawing conclusions.
- Temperature scaling can improve aggregate calibration on similar data but cannot make out-of-distribution confidence trustworthy.
- Weighted sampling mitigates imbalance during optimization; it does not correct deployment prevalence or unequal error costs.
- ImageNet transfer may encode irrelevant features. Inspect false positives, false negatives, and Grad-CAM maps before deployment.
- SQLite and a single-process local tracker are appropriate for a portfolio demo, not high-throughput multi-worker production.
- No production accuracy, safety, uptime, or quality guarantee is claimed.

## Suggested portfolio narrative

Start with the baseline as an honest lower bound, compare the transfer model under the same split, inspect false-negative Grad-CAMs, discuss calibration and class-specific recall, and state the exact limits above. The interesting engineering result is the reproducible path from raw images to a testable service—not a single headline accuracy number.
