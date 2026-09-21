from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from inspectai.config import IMAGE_SIZE

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_transforms(train: bool, image_size: int = IMAGE_SIZE):
    operations: list[Callable] = [transforms.Resize((image_size, image_size))]
    if train:
        operations += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(5)]
    operations += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(operations)


class BinaryDefectDataset(Dataset):
    """Image-folder dataset with split/normal and split/defective directories."""

    class_names = ["normal", "defective"]

    def __init__(self, root: str | Path, split: str, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform or build_transforms(split == "train")
        self.samples: list[tuple[Path, int]] = []
        for label, name in enumerate(self.class_names):
            folder = self.root / split / name
            if folder.exists():
                self.samples.extend(
                    (path, label)
                    for path in sorted(folder.rglob("*"))
                    if path.suffix.lower() in VALID_EXTENSIONS
                )
        if not self.samples:
            raise ValueError(f"No images found under {self.root / split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label, str(path)


def validate_dataset(root: str | Path) -> dict[str, dict[str, int]]:
    root = Path(root)
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for class_name in BinaryDefectDataset.class_names:
            folder = root / split / class_name
            count = sum(
                p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
                for p in folder.rglob("*")
            ) if folder.exists() else 0
            counts[split][class_name] = count
            if count == 0:
                raise ValueError(f"Missing images for {split}/{class_name} in {root}")
    return counts

