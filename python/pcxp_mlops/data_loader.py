"""Dataset and image-loading utilities."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64

from .config import DEFAULT_CLASS_NAMES, DEFAULT_TARGET_SIZE, ProjectPaths


def build_transforms(target_size: int = DEFAULT_TARGET_SIZE):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class RSNAPneumoniaDataset:
    """Custom dataset reading images by patientId from CSV metadata."""

    def __init__(self, csv_path: str | Path, images_dir: str | Path, transform=None):
        import pandas as pd
        self.images_dir = Path(images_dir)
        self.transform = transform
        df = pd.read_csv(csv_path)
        self.df = df.drop_duplicates(subset=["patientId"]).reset_index(drop=True)
        self.classes = list(DEFAULT_CLASS_NAMES)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        from PIL import Image
        row = self.df.iloc[idx]
        target = int(row["Target"])
        image_path = self.images_dir / f"{row['patientId']}.png"
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target

    def get_metadata_array(self, indices: list[int] | None = None):
        import numpy as np
        subset = self.df if indices is None else self.df.iloc[indices]
        ages = subset["age"].astype(float).values / 100.0
        sexes = (subset["sex"] == "F").astype(float).values
        positions = (subset["position"] == "PA").astype(float).values
        return np.column_stack([ages, sexes, positions])


class RSNAPneumoniaEvalDataset:
    """Dataset for the unlabeled test set (no Target column)."""

    def __init__(self, csv_path: str | Path, images_dir: str | Path, transform=None):
        import pandas as pd
        self.images_dir = Path(images_dir)
        self.transform = transform
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)
        self.classes = list(DEFAULT_CLASS_NAMES)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        from PIL import Image
        row = self.df.iloc[idx]
        image_path = self.images_dir / f"{row['patientId']}.png"
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

    def get_metadata_array(self, indices: list[int] | None = None):
        import numpy as np
        subset = self.df if indices is None else self.df.iloc[indices]
        ages = subset["age"].astype(float).values / 100.0
        sexes = (subset["sex"] == "F").astype(float).values
        positions = (subset["position"] == "PA").astype(float).values
        return np.column_stack([ages, sexes, positions])


def load_datasets(paths: ProjectPaths, target_size: int = DEFAULT_TARGET_SIZE):
    """Load train/validation datasets from the RSNA metadata CSV."""
    transform = build_transforms(target_size)
    full_dataset = RSNAPneumoniaDataset(paths.train_metadata_csv, paths.images_dir, transform)
    import torch
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator,
    )
    train_dataset.classes = full_dataset.classes
    val_dataset.classes = full_dataset.classes
    return train_dataset, val_dataset


def count_images_by_label(test_dir: str | Path) -> list[tuple[str, int]]:
    test_dir = Path(test_dir)
    counts: list[tuple[str, int]] = []
    total = 0
    for label_dir in sorted(test_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        image_count = sum(1 for entry in label_dir.iterdir() if entry.is_file())
        counts.append((label_dir.name, image_count))
        total += image_count
    counts.append(("Total", total))
    return counts


def load_image_from_path(image_path: str | Path):
    from PIL import Image
    image = Image.open(image_path)
    return image.convert("RGB")


def load_image_from_base64(encoded_image: str):
    from PIL import Image
    raw_bytes = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(raw_bytes))
    return image.convert("RGB")
