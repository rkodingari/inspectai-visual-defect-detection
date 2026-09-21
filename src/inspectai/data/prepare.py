from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

from inspectai.data.dataset import VALID_EXTENSIONS, validate_dataset

MVTEC_CATEGORY_URL = "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_anomaly_detection/{category}.tar.xz"
MVTEC_DATASET_PAGE = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"


def _make_image(path: Path, defective: bool, rng: np.random.Generator, size: int = 128) -> None:
    base = rng.normal(165, 9, (size, size, 3)).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((38, 13, 90, 115), radius=18, outline=(90, 100, 105), width=4)
    draw.rectangle((51, 6, 77, 18), fill=(120, 130, 135))
    if defective:
        x, y = rng.integers(47, 75), rng.integers(30, 95)
        draw.ellipse((x - 8, y - 5, x + 8, y + 5), fill=(45, 35, 30))
        draw.line((x - 14, y + 9, x + 14, y - 10), fill=(20, 20, 20), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def generate_synthetic(output: Path, seed: int = 42, force: bool = False) -> dict:
    if output.exists() and force:
        shutil.rmtree(output)
    rng = np.random.default_rng(seed)
    split_sizes = {"train": 24, "val": 8, "test": 8}
    for split, per_class in split_sizes.items():
        for class_name in ("normal", "defective"):
            for index in range(per_class):
                _make_image(
                    output / split / class_name / f"{class_name}_{index:03d}.png",
                    defective=class_name == "defective",
                    rng=rng,
                )
    return validate_dataset(output)


def download_mvtec(raw_dir: Path, category: str) -> Path:
    archive = raw_dir / f"{category}.tar.xz"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading MVTec AD category '{category}' to {archive}...")
        try:
            urllib.request.urlretrieve(MVTEC_CATEGORY_URL.format(category=category), archive)
        except (urllib.error.URLError, urllib.error.HTTPError) as error:
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                "Automatic MVTec download failed. Download the category archive from "
                f"{MVTEC_DATASET_PAGE}, extract it, and rerun with --source /path/to/root."
            ) from error
    extracted = raw_dir / f"mvtec_{category}"
    if not extracted.exists():
        extracted.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            destination = extracted.resolve()
            members = tar.getmembers()
            if any(not (destination / member.name).resolve().is_relative_to(destination)
                   for member in members):
                raise ValueError("Unsafe path found in dataset archive")
            tar.extractall(extracted, members=members)
    return extracted


def _copy_files(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        # Prefix anomaly type to avoid repeated names across MVTec directories.
        target = destination / f"{source.parent.name}_{source.name}"
        shutil.copy2(source, target)


def prepare_mvtec(source_root: Path, output: Path, category: str = "bottle", seed: int = 42,
                  quick: bool = False, force: bool = False) -> dict:
    category_root = source_root / category
    if not category_root.exists():
        # Accommodate archives that contain one additional directory level.
        matches = list(source_root.rglob(f"{category}/train/good"))
        if not matches:
            raise FileNotFoundError(f"Could not find MVTec category '{category}' in {source_root}")
        category_root = matches[0].parents[1]
    if output.exists() and force:
        shutil.rmtree(output)
    good = sorted(p for p in (category_root / "train" / "good").glob("*") if p.suffix.lower() in VALID_EXTENSIONS)
    defects = sorted(p for p in (category_root / "test").glob("*/*") if p.parent.name != "good" and p.suffix.lower() in VALID_EXTENSIONS)
    test_good = sorted(p for p in (category_root / "test" / "good").glob("*") if p.suffix.lower() in VALID_EXTENSIONS)
    if quick:
        good, defects, test_good = good[:40], defects[:40], test_good[:20]
    train_good, val_good = train_test_split(good, test_size=0.2, random_state=seed)
    # MVTec has no defective training images. For supervised binary classification,
    # split known defect examples into train/validation/test and document this choice.
    train_def, remaining_def = train_test_split(defects, test_size=0.4, random_state=seed)
    val_def, test_def = train_test_split(remaining_def, test_size=0.5, random_state=seed)
    val_test_good = val_good + test_good
    val_normal, test_normal = train_test_split(val_test_good, test_size=0.5, random_state=seed)
    groups = {
        ("train", "normal"): train_good, ("train", "defective"): train_def,
        ("val", "normal"): val_normal, ("val", "defective"): val_def,
        ("test", "normal"): test_normal, ("test", "defective"): test_def,
    }
    for (split, label), files in groups.items():
        _copy_files(files, output / split / label)
    counts = validate_dataset(output)
    (output / "dataset_metadata.json").write_text(json.dumps({
        "source": "MVTec AD", "category": category, "seed": seed,
        "quick": quick, "counts": counts,
        "note": "Defect test examples were split for supervised training; this is a portfolio classification setup, not the canonical unsupervised MVTec protocol."
    }, indent=2))
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare InspectAI data")
    parser.add_argument("--dataset", choices=("synthetic", "mvtec"), default="synthetic")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--source", type=Path, help="Existing extracted MVTec root")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path("data/processed") / ("synthetic" if args.dataset == "synthetic" else f"mvtec_{args.category}")
    if args.dataset == "synthetic":
        counts = generate_synthetic(output, args.seed, args.force)
    else:
        source = args.source or download_mvtec(args.raw_dir, args.category)
        counts = prepare_mvtec(source, output, args.category, args.seed, args.quick, args.force)
    print(json.dumps({"output": str(output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
