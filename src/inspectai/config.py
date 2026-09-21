from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(os.getenv("INSPECTAI_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
DEFAULT_CHECKPOINT = Path(os.getenv("INSPECTAI_MODEL_PATH", ARTIFACT_DIR / "best_model.pt"))
IMAGE_SIZE = 224
CLASS_NAMES = ["normal", "defective"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

