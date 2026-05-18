"""Inference service shared by the API and manual prediction flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .config import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TARGET_SIZE,
    DEFAULT_THRESHOLD,
    ProjectPaths,
    get_paths,
)
from .data_loader import build_transforms, load_image_from_base64, load_image_from_path
from .metadata import load_metadata
from .model import build_resnet18


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    predicted_index: int
    probability: float
    threshold: float
    model_version: str


class InferenceService:
    """Lazy-loading image inference service for the REST API."""

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        target_size: int = DEFAULT_TARGET_SIZE,
        default_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.paths = paths or get_paths()
        self.target_size = target_size
        self.default_threshold = default_threshold
        self._model = None
        self._device = None
        self._class_names: list[str] | None = None
        self._metadata: dict | None = None

    def _load_class_names(self) -> list[str]:
        if self._class_names is not None:
            return self._class_names
        if self.paths.classes_path.exists():
            with self.paths.classes_path.open("r", encoding="utf-8") as handle:
                self._class_names = json.load(handle)
                return self._class_names
        if self.paths.train_dir.exists():
            self._class_names = sorted(
                entry.name for entry in self.paths.train_dir.iterdir() if entry.is_dir()
            )
            if self._class_names:
                return self._class_names
        self._class_names = list(DEFAULT_CLASS_NAMES)
        return self._class_names

    def _load_metadata(self) -> dict:
        if self._metadata is None:
            self._metadata = load_metadata(self.paths)
        return self._metadata

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch

        if not self.paths.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.paths.model_path}")

        class_names = self._load_class_names()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_resnet18(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(torch.load(self.paths.model_path, map_location=self._device))
        model.to(self._device)
        model.eval()
        self._model = model

    def health(self) -> dict:
        """Return serving health and model metadata."""
        metadata = self._load_metadata()
        return {
            "status": "ok" if self.paths.model_path.exists() else "missing-model",
            "model_version": metadata.get("model_version", DEFAULT_MODEL_VERSION),
            "model_path": str(self.paths.model_path),
        }

    def model_info(self) -> dict:
        """Return model metadata and file-level serving information."""
        metadata = dict(self._load_metadata())
        metadata["model_path"] = str(self.paths.model_path)
        metadata["has_weights"] = self.paths.model_path.exists()
        metadata["class_names"] = self._load_class_names()
        return metadata

    def predict(
        self,
        image_path: str | None = None,
        image_base64: str | None = None,
        threshold: float | None = None,
    ) -> PredictionResult:
        """Run inference against an image path or a base64-encoded image."""
        import torch

        if not image_path and not image_base64:
            raise ValueError("Provide either image_path or image_base64.")
        if image_path and image_base64:
            raise ValueError("Provide only one of image_path or image_base64.")

        self._load_model()
        transform = build_transforms(self.target_size)
        active_threshold = threshold if threshold is not None else self.default_threshold

        if image_path:
            image = load_image_from_path(Path(image_path))
        else:
            image = load_image_from_base64(image_base64 or "")

        tensor = transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

        predicted_index = 1 if probabilities[1] >= active_threshold else 0
        class_names = self._load_class_names()
        metadata = self._load_metadata()

        return PredictionResult(
            predicted_class=class_names[predicted_index],
            predicted_index=predicted_index,
            probability=float(probabilities[predicted_index]),
            threshold=active_threshold,
            model_version=metadata.get("model_version", DEFAULT_MODEL_VERSION),
        )
