"""Dataset and image-loading utilities."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64

from .config import DEFAULT_TARGET_SIZE, ProjectPaths


def build_transforms(target_size: int = DEFAULT_TARGET_SIZE):
    """Create the image preprocessing pipeline shared by train/eval/inference."""
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


def load_datasets(paths: ProjectPaths, target_size: int = DEFAULT_TARGET_SIZE):
    """Load train and test datasets from the standard project layout."""
    from torchvision import datasets

    transform = build_transforms(target_size)
    train_dataset = datasets.ImageFolder(root=str(paths.train_dir), transform=transform)
    test_dataset = datasets.ImageFolder(root=str(paths.test_dir), transform=transform)
    return train_dataset, test_dataset


def count_images_by_label(test_dir: str | Path) -> list[tuple[str, int]]:
    """Count test images per label directory and append the total."""
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


def load_image_from_path(image_path: str | Path) -> Image.Image:
    """Load an RGB image from disk."""
    from PIL import Image

    image = Image.open(image_path)
    return image.convert("RGB")


def load_image_from_base64(encoded_image: str) -> Image.Image:
    """Load an RGB image from a base64 string."""
    from PIL import Image

    raw_bytes = base64.b64decode(encoded_image)
    image = Image.open(BytesIO(raw_bytes))
    return image.convert("RGB")
