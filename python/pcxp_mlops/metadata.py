"""Model metadata persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from .config import (
    BASE_MODEL_CONFIGS,
    DEFAULT_CLASS_NAMES,
    DEFAULT_MODEL_VERSION,
    DEFAULT_THRESHOLD,
    ENSEMBLE_MODEL_VERSION,
    PROJECT_VERSION,
    ProjectPaths,
)


def default_metadata() -> dict:
    """Return the baseline metadata structure used by the API."""
    return {
        "project_version": PROJECT_VERSION,
        "model_name": "resnet18",
        "model_version": DEFAULT_MODEL_VERSION,
        "dataset": "RSNA pneumonia detection dataset",
        "threshold": DEFAULT_THRESHOLD,
        "class_names": DEFAULT_CLASS_NAMES,
        "trained_at": None,
        "metrics": {},
    }


def default_ensemble_metadata() -> dict:
    """Return the baseline metadata structure for the stacked ensemble."""
    return {
        "project_version": PROJECT_VERSION,
        "model_name": "stacking_ensemble",
        "model_version": ENSEMBLE_MODEL_VERSION,
        "dataset": "RSNA pneumonia detection dataset",
        "threshold": DEFAULT_THRESHOLD,
        "base_models": list(BASE_MODEL_CONFIGS.keys()),
        "class_names": DEFAULT_CLASS_NAMES,
        "trained_at": None,
        "metrics": {},
    }


def load_metadata(paths: ProjectPaths) -> dict:
    """Load metadata if available, otherwise return the default structure."""
    if paths.metadata_path.exists():
        with paths.metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return default_metadata()


def load_ensemble_metadata(paths: ProjectPaths) -> dict:
    """Load ensemble metadata if available, otherwise return the default."""
    meta_path = paths.ensemble_dir / "metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return default_ensemble_metadata()


def save_metadata(paths: ProjectPaths, metadata: dict) -> None:
    """Persist model metadata to disk."""
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    with paths.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def save_ensemble_metadata(paths: ProjectPaths, metadata: dict) -> None:
    """Persist ensemble metadata to disk."""
    paths.ensemble_dir.mkdir(parents=True, exist_ok=True)
    meta_path = paths.ensemble_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def stamp_training_time(metadata: dict) -> dict:
    """Attach the current UTC training time to the metadata payload."""
    payload = dict(metadata)
    payload["trained_at"] = datetime.now(timezone.utc).isoformat()
    return payload
