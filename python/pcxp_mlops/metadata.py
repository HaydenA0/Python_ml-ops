"""Model metadata persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from .config import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_MODEL_VERSION,
    DEFAULT_THRESHOLD,
    PROJECT_VERSION,
    ProjectPaths,
)


def default_metadata() -> dict:
    """Return the baseline metadata structure used by the API."""
    return {
        "project_version": PROJECT_VERSION,
        "model_name": "resnet18",
        "model_version": DEFAULT_MODEL_VERSION,
        "dataset": "PCXP chest X-ray pneumonia dataset",
        "threshold": DEFAULT_THRESHOLD,
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


def save_metadata(paths: ProjectPaths, metadata: dict) -> None:
    """Persist model metadata to disk."""
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    with paths.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def stamp_training_time(metadata: dict) -> dict:
    """Attach the current UTC training time to the metadata payload."""
    payload = dict(metadata)
    payload["trained_at"] = datetime.now(timezone.utc).isoformat()
    return payload
