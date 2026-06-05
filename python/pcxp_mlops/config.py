"""Configuration helpers for training, evaluation, and serving."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

PROJECT_VERSION = "0.1.0"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MODEL_VERSION = "pcxp-resnet18-v1"
DEFAULT_CLASS_NAMES = ["No Lung Opacity", "Lung Opacity"]
DEFAULT_TARGET_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3

CLINICAL_FEATURES = ["age", "sex", "position"]

ENSEMBLE_MODEL_VERSION = "pcxp-ensemble-v1"
BASE_MODEL_CONFIGS = {
    "resnet18": {"builder": "resnet18", "pretrained": True, "lr": 1e-3, "epochs": 5},
    "tiny_cnn": {"builder": "tiny_cnn", "pretrained": False, "lr": 1e-3, "epochs": 8},
    "resnet18_low_lr": {"builder": "resnet18", "pretrained": True, "lr": 1e-4, "epochs": 5},
}


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by the project."""

    project_root: Path
    data_dir: Path
    images_dir: Path
    train_metadata_csv: Path
    test_metadata_csv: Path
    model_dir: Path
    model_path: Path
    classes_path: Path
    metadata_path: Path
    ensemble_dir: Path

    def as_dict(self) -> dict[str, str]:
        """Return string paths for compatibility with legacy code/tests."""
        return {
            "current_dir": str(self.project_root),
            "data_dir": str(self.data_dir),
            "images_dir": str(self.images_dir),
            "train_metadata_csv": str(self.train_metadata_csv),
            "test_metadata_csv": str(self.test_metadata_csv),
            "model_dir": str(self.model_dir),
            "model_path": str(self.model_path),
            "classes_path": str(self.classes_path),
            "metadata_path": str(self.metadata_path),
            "ensemble_dir": str(self.ensemble_dir),
        }


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_paths(project_root: str | os.PathLike[str] | None = None) -> ProjectPaths:
    """Build absolute project paths from the repository root."""
    root = Path(project_root) if project_root is not None else _default_root()
    root = root.resolve()
    data_dir = root / "data"
    model_dir = root / "models"
    return ProjectPaths(
        project_root=root,
        data_dir=data_dir,
        images_dir=data_dir / "Training" / "Images",
        train_metadata_csv=data_dir / "stage2_train_metadata.csv",
        test_metadata_csv=data_dir / "stage2_test_metadata.csv",
        model_dir=model_dir,
        model_path=model_dir / "model.pth",
        classes_path=model_dir / "classes.json",
        metadata_path=model_dir / "metadata.json",
        ensemble_dir=model_dir / "ensemble",
    )
