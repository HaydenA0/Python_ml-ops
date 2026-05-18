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

from .config import DEFAULT_BATCH_SIZE, DEFAULT_TARGET_SIZE, DEFAULT_THRESHOLD, get_paths
from .data_loader import count_images_by_label, load_datasets
from .metadata import load_metadata, save_metadata
from .metrics import apply_threshold, recall_from_confusion_matrix
from .model import build_resnet18


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
            mlflow.log_params({
                "target_size": target_size,
                "batch_size": batch_size,
                "threshold": threshold,
            })
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = Path(tmpdir) / "confusion_matrix.txt"
                artifact_path.write_text(str(cm), encoding="utf-8")
                mlflow.log_artifact(str(artifact_path))

    print(count_images_by_label(paths.test_dir))
    print("Confusion matrix:")
    print(cm)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"{metric_name}: {metric_value}")

    return {"confusion_matrix": cm.tolist(), "metrics": metrics}


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()
