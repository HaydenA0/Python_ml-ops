"""Evaluation pipeline with optional MLflow logging."""

from __future__ import annotations

from pathlib import Path
import tempfile

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    BASE_MODEL_CONFIGS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TARGET_SIZE,
    DEFAULT_THRESHOLD,
    ENSEMBLE_MODEL_VERSION,
    get_paths,
)
from .data_loader import count_images_by_label, load_datasets
from .metadata import (
    load_ensemble_metadata,
    load_metadata,
    save_ensemble_metadata,
    save_metadata,
)
from .metrics import apply_threshold, recall_from_confusion_matrix
from .model import BaseModelWrapper, StackingEnsemble, build_resnet18, build_tiny_cnn


def _maybe_get_mlflow():
    try:
        import mlflow
    except ModuleNotFoundError:
        return None
    return mlflow


def evaluate_model(
    target_size: int = DEFAULT_TARGET_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Evaluate the saved classifier and update serving metadata."""
    import torch
    import torch.nn.functional as functional

    paths = get_paths()
    _, test_dataset = load_datasets(paths, target_size)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(test_dataset.classes), pretrained=False)
    model.load_state_dict(torch.load(paths.model_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels: list[int] = []
    all_probs: list[list[float]] = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = functional.softmax(outputs, dim=1).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().tolist())

    all_preds = apply_threshold(all_probs, threshold)
    positive_probs = [class_probs[1] for class_probs in all_probs]
    cm = confusion_matrix(all_labels, all_preds)
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, positive_probs),
        "threshold": threshold,
        "positive_class_recall_from_confusion_matrix": recall_from_confusion_matrix(cm),
    }

    metadata = load_metadata(paths)
    metadata["metrics"] = metrics
    metadata["class_names"] = test_dataset.classes
    save_metadata(paths, metadata)

    mlflow = _maybe_get_mlflow()
    if mlflow is not None:
        mlflow.set_experiment("Pneumonia_Model_Evaluation")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "target_size": target_size,
                    "batch_size": batch_size,
                    "threshold": threshold,
                }
            )
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = Path(tmpdir) / "confusion_matrix.txt"
                artifact_path.write_text(str(cm), encoding="utf-8")
                mlflow.log_artifact(str(artifact_path))

    print(f"Evaluated {len(all_labels)} samples")
    print("Confusion matrix:")
    print(cm)
    for metric_name, metric_value in metrics.items():
        print(
            f"{metric_name}: {metric_value:.4f}"
            if isinstance(metric_value, float)
            else f"{metric_name}: {metric_value}"
        )

    return {"confusion_matrix": cm.tolist(), "metrics": metrics}


def _load_ensemble_models(paths, device, num_classes):
    import joblib
    import torch
    from .model import build_resnet18, build_tiny_cnn

    meta_path = paths.ensemble_dir / "meta_model.pkl"
    meta_model = joblib.load(meta_path) if meta_path.exists() else None

    wrappers = []
    for name in BASE_MODEL_CONFIGS:
        model_path = paths.ensemble_dir / f"{name}.pth"
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
        model.load_state_dict(torch.load(model_path, map_location=device))
        wrappers.append(BaseModelWrapper(model, name, device))

    return StackingEnsemble(wrappers, meta_model=meta_model)


def evaluate_ensemble(
    target_size: int = DEFAULT_TARGET_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    import torch
    from .data_loader import RSNAPneumoniaDataset, build_transforms

    paths = get_paths()
    full_dataset = RSNAPneumoniaDataset(
        paths.train_metadata_csv,
        paths.images_dir,
        build_transforms(target_size),
    )
    _, val_dataset = load_datasets(paths, target_size)
    val_indices = val_dataset.indices
    val_metadata = full_dataset.get_metadata_array(val_indices)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = _load_ensemble_models(paths, device, len(val_dataset.classes))

    all_probs: list[float] = []
    all_labels: list[int] = []
    meta_offset = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            b = images.size(0)
            batch_metadata = val_metadata[meta_offset : meta_offset + b]
            meta_offset += b
            batch_probs = ensemble.predict(
                images, run_parallel=False, metadata=batch_metadata
            )
            all_probs.extend(batch_probs)
            all_labels.extend(labels.cpu().tolist())

    all_preds = [1 if p >= threshold else 0 for p in all_probs]

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    cm = confusion_matrix(all_labels, all_preds)
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs),
        "threshold": threshold,
        "positive_class_recall_from_confusion_matrix": recall_from_confusion_matrix(cm),
    }

    metadata = load_ensemble_metadata(paths)
    metadata["metrics"] = metrics
    metadata["class_names"] = val_dataset.classes
    save_ensemble_metadata(paths, metadata)

    print(f"Ensemble evaluation ({ENSEMBLE_MODEL_VERSION}):")
    print("Confusion matrix:")
    print(cm)
    for metric_name, metric_value in metrics.items():
        print(
            f"{metric_name}: {metric_value:.4f}"
            if isinstance(metric_value, float)
            else f"{metric_name}: {metric_value}"
        )

    mlflow = _maybe_get_mlflow()
    if mlflow is not None:
        mlflow.set_experiment("Pneumonia_Ensemble_Evaluation")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "target_size": target_size,
                    "batch_size": batch_size,
                    "threshold": threshold,
                    "ensemble_version": ENSEMBLE_MODEL_VERSION,
                }
            )
            mlflow.log_metrics(metrics)

    return {"confusion_matrix": cm.tolist(), "metrics": metrics}


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()
