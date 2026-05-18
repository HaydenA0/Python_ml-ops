"""Configuration helpers for training, evaluation, and serving."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

PROJECT_VERSION = "0.1.0"
DEFAULT_THRESHOLD = 0.87
DEFAULT_MODEL_VERSION = "pcxp-resnet18-v1"
DEFAULT_CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEFAULT_TARGET_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by the project."""

    project_root: Path
    data_dir: Path
    train_dir: Path
    test_dir: Path
    model_dir: Path
    model_path: Path
    classes_path: Path
    metadata_path: Path

    def as_dict(self) -> dict[str, str]:
        """Return string paths for compatibility with legacy code/tests."""
        return {
            "current_dir": str(self.project_root),
            "data_dir": str(self.data_dir),
            "train_dir": str(self.train_dir),
            "test_dir": str(self.test_dir),
            "model_dir": str(self.model_dir),
            "model_path": str(self.model_path),
            "classes_path": str(self.classes_path),
            "metadata_path": str(self.metadata_path),
        }


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_paths(project_root: str | os.PathLike[str] | None = None) -> ProjectPaths:
    """Build absolute project paths from the repository root."""
    root = Path(project_root) if project_root is not None else _default_root()
    root = root.resolve()
    data_dir = root / "data" / "PCXP"
    model_dir = root / "models"
    return ProjectPaths(
        project_root=root,
        data_dir=data_dir,
        train_dir=data_dir / "train",
        test_dir=data_dir / "test",
        model_dir=model_dir,
        model_path=model_dir / "model.pth",
        classes_path=model_dir / "classes.json",
        metadata_path=model_dir / "metadata.json",
    )
