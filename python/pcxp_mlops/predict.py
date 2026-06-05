"""Inference service shared by the API and manual prediction flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .config import (
    BASE_MODEL_CONFIGS,
    DEFAULT_CLASS_NAMES,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TARGET_SIZE,
    DEFAULT_THRESHOLD,
    ENSEMBLE_MODEL_VERSION,
    ProjectPaths,
    get_paths,
)
from .data_loader import build_transforms, load_image_from_base64, load_image_from_path
from .metadata import load_ensemble_metadata, load_metadata
from .model import BaseModelWrapper, StackingEnsemble, build_resnet18, build_tiny_cnn


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    predicted_index: int
    probability: float
    threshold: float
    model_version: str
    latency_ms: float = 0.0
    device: str = "unknown"
    base_model_probabilities: dict[str, float] | None = None
    clinical_metadata: dict | None = None
    preprocessing: dict | None = None


def _device_label(device) -> str:
    import torch
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"GPU ({name})"
    return "CPU"


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
        import time

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
        start = time.perf_counter()
        with torch.no_grad():
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
        latency_ms = (time.perf_counter() - start) * 1000

        predicted_index = 1 if probabilities[1] >= active_threshold else 0
        class_names = self._load_class_names()
        metadata = self._load_metadata()

        return PredictionResult(
            predicted_class=class_names[predicted_index],
            predicted_index=predicted_index,
            probability=float(probabilities[predicted_index]),
            threshold=active_threshold,
            model_version=metadata.get("model_version", DEFAULT_MODEL_VERSION),
            latency_ms=round(latency_ms, 1),
            device=_device_label(self._device),
            preprocessing={"resized_to": f"{self.target_size}x{self.target_size}"},
        )


class EnsembleInferenceService:
    """Lazy-loading ensemble inference service using stacked base models + meta-learner."""

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        target_size: int = DEFAULT_TARGET_SIZE,
        default_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.paths = paths or get_paths()
        self.target_size = target_size
        self.default_threshold = default_threshold
        self._ensemble: StackingEnsemble | None = None
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
        self._class_names = list(DEFAULT_CLASS_NAMES)
        return self._class_names

    def _load_metadata(self) -> dict:
        if self._metadata is None:
            self._metadata = load_ensemble_metadata(self.paths)
        return self._metadata

    def _load_ensemble(self) -> None:
        if self._ensemble is not None:
            return
        import torch
        import joblib

        if not self.paths.ensemble_dir.exists():
            raise FileNotFoundError(f"Ensemble directory not found at {self.paths.ensemble_dir}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_names = self._load_class_names()
        num_classes = len(class_names)

        meta_path = self.paths.ensemble_dir / "meta_model.pkl"
        meta_model = joblib.load(meta_path) if meta_path.exists() else None

        base_wrappers: list[BaseModelWrapper] = []
        expected = list(BASE_MODEL_CONFIGS.keys())
        for name in expected:
            model_path = self.paths.ensemble_dir / f"{name}.pth"
            if not model_path.exists():
                if meta_model is not None:
                    msg = f"Base model {name} required by meta-learner not found at {model_path}"
                    raise FileNotFoundError(msg)
                continue
            cfg = BASE_MODEL_CONFIGS[name]
            if cfg["builder"] == "resnet18":
                model = build_resnet18(num_classes, pretrained=False)
            elif cfg["builder"] == "tiny_cnn":
                model = build_tiny_cnn(num_classes)
            else:
                continue
            model.load_state_dict(torch.load(model_path, map_location=self._device))
            base_wrappers.append(BaseModelWrapper(model, name, self._device))

        self._ensemble = StackingEnsemble(base_wrappers, meta_model=meta_model)

    def health(self) -> dict:
        metadata = self._load_metadata()
        return {
            "status": "ok" if self.paths.ensemble_dir.exists() else "missing-ensemble",
            "model_version": metadata.get("model_version", ENSEMBLE_MODEL_VERSION),
            "ensemble_path": str(self.paths.ensemble_dir),
        }

    def model_info(self) -> dict:
        metadata = dict(self._load_metadata())
        metadata["ensemble_path"] = str(self.paths.ensemble_dir)
        metadata["has_base_models"] = (
            (self.paths.ensemble_dir / "resnet18.pth").exists()
            and (self.paths.ensemble_dir / "tiny_cnn.pth").exists()
            and (self.paths.ensemble_dir / "resnet18_low_lr.pth").exists()
        )
        metadata["has_meta_model"] = (self.paths.ensemble_dir / "meta_model.pkl").exists()
        metadata["class_names"] = self._load_class_names()
        return metadata

    def _build_clinical_metadata(self, age: int | None, sex: str | None, position: str | None):
        import numpy as np
        age_norm = float(age or 50) / 100.0
        sex_enc = 0.0 if (sex or "M") == "M" else 1.0
        pos_enc = 0.0 if (position or "AP") == "AP" else 1.0
        return np.array([[age_norm, sex_enc, pos_enc]])

    def predict(
        self,
        image_path: str | None = None,
        image_base64: str | None = None,
        threshold: float | None = None,
        age: int | None = None,
        sex: str | None = None,
        position: str | None = None,
    ) -> PredictionResult:
        import torch
        import time

        if not image_path and not image_base64:
            raise ValueError("Provide either image_path or image_base64.")
        if image_path and image_base64:
            raise ValueError("Provide only one of image_path or image_base64.")

        self._load_ensemble()
        transform = build_transforms(self.target_size)
        active_threshold = threshold if threshold is not None else self.default_threshold

        if image_path:
            image = load_image_from_path(Path(image_path))
        else:
            image = load_image_from_base64(image_base64 or "")

        tensor = transform(image).unsqueeze(0).to(self._device)
        clinical = self._build_clinical_metadata(age, sex, position)

        start = time.perf_counter()
        import numpy as np
        base_dict = self._ensemble.get_base_predictions(tensor, run_parallel=True)
        base_probs = {name: float(probs[0]) for name, probs in base_dict.items()}
        meta_input = np.column_stack([base_dict[n] for n in BASE_MODEL_CONFIGS])
        if clinical is not None:
            meta_input = np.column_stack([meta_input, clinical])
        meta_model = self._ensemble.meta_model
        if meta_model is not None and hasattr(meta_model, "predict_proba"):
            probability = float(meta_model.predict_proba(meta_input)[0, 1])
        elif meta_model is not None:
            probability = float(meta_model.predict(meta_input)[0])
        elif self._ensemble._weights is not None:
            w = self._ensemble._weights
            probability = sum(wi * base_probs[n] for n, wi in zip(BASE_MODEL_CONFIGS, w)) / (sum(w) or 1.0)
        else:
            probability = sum(base_probs.values()) / max(len(base_probs), 1)
        latency_ms = (time.perf_counter() - start) * 1000

        predicted_index = 1 if probability >= active_threshold else 0
        class_names = self._load_class_names()
        metadata = self._load_metadata()

        return PredictionResult(
            predicted_class=class_names[predicted_index],
            predicted_index=predicted_index,
            probability=probability,
            threshold=active_threshold,
            model_version=metadata.get("model_version", ENSEMBLE_MODEL_VERSION),
            latency_ms=round(latency_ms, 1),
            device=_device_label(self._device),
            base_model_probabilities=base_probs,
            clinical_metadata={
                "age": age,
                "age_norm": round(float(age or 50) / 100.0, 4),
                "sex": sex or "M",
                "sex_enc": 0 if (sex or "M") == "M" else 1,
                "position": position or "AP",
                "pos_enc": 0 if (position or "AP") == "AP" else 1,
            },
            preprocessing={"resized_to": f"{self.target_size}x{self.target_size}"},
        )
