"""Training pipeline for the PCXP classifier."""

from __future__ import annotations

import json
import time

from .config import (
    BASE_MODEL_CONFIGS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TARGET_SIZE,
    ENSEMBLE_MODEL_VERSION,
    get_paths,
)
from .data_loader import load_datasets
from .metadata import (
    default_ensemble_metadata,
    default_metadata,
    save_ensemble_metadata,
    save_metadata,
    stamp_training_time,
)
from .model import build_resnet18, build_tiny_cnn


def _compute_accuracy(model, loader, device):
    import torch
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def _run_training(model, train_loader, device, epochs, lr, model_name=""):
    import torch
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    total_start = time.time()
    epoch = 0
    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_loss = running_loss / max(len(train_loader), 1)
            acc = correct / max(total, 1)
            epoch_time = time.time() - epoch_start
            print(f"  {model_name} | Epoch {epoch + 1:2d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.4f} | "
                  f"{epoch_time:.1f}s")
    except KeyboardInterrupt:
        print(f"\n  {model_name} | Interrupted — saving partial progress at epoch {epoch + 1}/{epochs}.")

    total_time = time.time() - total_start
    print(f"  {model_name} | Total: {total_time:.1f}s ({total_time/epochs:.1f}s/epoch)")


def _make_base_model(builder_key, num_classes):
    if builder_key == "resnet18":
        return build_resnet18(num_classes, pretrained=True)
    elif builder_key == "tiny_cnn":
        return build_tiny_cnn(num_classes)
    msg = f"Unknown builder: {builder_key}"
    raise ValueError(msg)


def train_model(
    target_size: int = DEFAULT_TARGET_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> None:
    """Train the project model and save weights plus serving metadata."""
    import torch

    paths = get_paths()
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, test_dataset = load_datasets(paths, target_size)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(train_dataset.classes), pretrained=True)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if paths.model_path.exists():
        print("Model found, loading existing weights before training...")
        model.load_state_dict(torch.load(paths.model_path, map_location=device))

    print("Training...\nPress Ctrl+C at any time to save progress and stop.")
    total_start = time.time()
    epoch = 0
    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for images, labels in train_dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)

            avg_train_loss = train_loss / max(len(train_dataloader), 1)
            avg_test_loss = test_loss / max(len(test_dataloader), 1)
            train_acc = train_correct / max(train_total, 1)
            test_acc = test_correct / max(test_total, 1)
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1:2d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_test_loss:.4f} Acc: {test_acc:.4f} | "
                f"{epoch_time:.1f}s"
            )
    except KeyboardInterrupt:
        print(f"\nInterrupted — saving partial progress at epoch {epoch + 1}/{epochs}.")

    total_time = time.time() - total_start
    print(f"Total training time: {total_time:.1f}s ({total_time/epochs:.1f}s/epoch)")

    torch.save(model.state_dict(), paths.model_path)

    with paths.classes_path.open("w", encoding="utf-8") as handle:
        json.dump(train_dataset.classes, handle, indent=2)

    metadata = default_metadata()
    metadata["class_names"] = train_dataset.classes
    metadata["model_version"] = DEFAULT_MODEL_VERSION
    metadata = stamp_training_time(metadata)
    save_metadata(paths, metadata)
    print(f"Saved model to {paths.model_path}")


def train_base_models(
    target_size: int = DEFAULT_TARGET_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    import torch
    paths = get_paths()
    paths.ensemble_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, _ = load_datasets(paths, target_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_dataset.classes)

    n_models = len(BASE_MODEL_CONFIGS)
    total_start = time.time()
    for i, (name, cfg) in enumerate(BASE_MODEL_CONFIGS.items(), 1):
        print(f"\n[{i}/{n_models}] Training base model: {name}")
        print(f"  Architecture: {cfg['builder']} | LR: {cfg['lr']} | Epochs: {cfg['epochs']}")
        model_start = time.time()
        model = _make_base_model(cfg["builder"], num_classes)
        _run_training(model, train_loader, device, cfg["epochs"], cfg["lr"], name)
        save_path = paths.ensemble_dir / f"{name}.pth"
        torch.save(model.state_dict(), save_path)
        model_time = time.time() - model_start
        print(f"  Saved {name} ({model_time:.1f}s total)")
    total_time = time.time() - total_start
    print(f"\nAll {n_models} base models trained in {total_time:.1f}s")


def train_meta_learner(
    target_size: int = DEFAULT_TARGET_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split: float = 0.2,
) -> None:
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from torch.utils.data import random_split

    from .data_loader import RSNAPneumoniaDataset, build_transforms

    paths = get_paths()
    paths.ensemble_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = RSNAPneumoniaDataset(
        paths.train_metadata_csv, paths.images_dir, build_transforms(target_size),
    )
    num_classes = len(full_dataset.classes)

    generator = torch.Generator().manual_seed(42)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator,
    )
    val_metadata = full_dataset.get_metadata_array(val_subset.indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
    )

    print(f"\nTraining base models on {train_size} samples (val: {val_size} samples)...")
    trained_wrappers: dict[str, torch.nn.Module] = {}
    n_models = len(BASE_MODEL_CONFIGS)
    meta_epochs = {name: max(cfg["epochs"] // 2, 2) for name, cfg in BASE_MODEL_CONFIGS.items()}
    for i, (name, cfg) in enumerate(BASE_MODEL_CONFIGS.items(), 1):
        epochs = meta_epochs[name]
        print(f"\n  [{i}/{n_models}] {name} ({cfg['builder']}, {epochs} epochs, LR={cfg['lr']})")
        model = _make_base_model(cfg["builder"], num_classes)
        _run_training(model, train_loader, device, epochs, cfg["lr"], name)
        trained_wrappers[name] = model

    print(f"\nGenerating validation predictions ({val_size} samples)...")
    all_preds: dict[str, list[float]] = {name: [] for name in BASE_MODEL_CONFIGS}
    all_labels: list[int] = []
    val_start = time.time()
    for images, labels in val_loader:
        images = images.to(device)
        all_labels.extend(labels.cpu().numpy())
        with torch.no_grad():
            for name, model in trained_wrappers.items():
                model.eval()
                model.to(device)
                probs = torch.softmax(model(images), dim=1)[:, 1].cpu().numpy()
                all_preds[name].extend(probs)
    print(f"  Validation predictions collected in {time.time() - val_start:.1f}s")

    prob_features = np.column_stack([all_preds[name] for name in BASE_MODEL_CONFIGS])
    meta_X = np.column_stack([prob_features, val_metadata])
    meta_y = np.array(all_labels)

    print(f"Training meta-learner (LogisticRegression, {meta_X.shape[1]} features)...")
    fit_start = time.time()
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_X, meta_y)
    fit_time = time.time() - fit_start

    meta_acc = meta_model.score(meta_X, meta_y)
    print(f"  Meta-learner trained in {fit_time:.1f}s")
    print(f"  Training accuracy: {meta_acc:.4f}")

    import joblib
    meta_path = paths.ensemble_dir / "meta_model.pkl"
    joblib.dump(meta_model, meta_path)
    print(f"Saved meta-learner to {meta_path}")

    metadata = default_ensemble_metadata()
    metadata["class_names"] = full_dataset.classes
    metadata["meta_learner_type"] = "LogisticRegression"
    metadata["val_samples"] = val_size
    metadata["clinical_features"] = ["age", "sex", "position"]
    metadata["meta_train_acc"] = float(round(meta_acc, 4))
    metadata = stamp_training_time(metadata)
    save_ensemble_metadata(paths, metadata)


def train_ensemble(
    target_size: int = DEFAULT_TARGET_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    ensemble_start = time.time()
    print("=" * 60)
    print("ENSEMBLE TRAINING")
    print("=" * 60)
    print(f"\nPhase 1/2: Training {len(BASE_MODEL_CONFIGS)} base models on full data...")
    train_base_models(target_size, batch_size)
    p1_time = time.time()
    print(f"\nPhase 2/2: Training meta-learner on held-out predictions...")
    train_meta_learner(target_size, batch_size, val_split=0.2)
    total_time = time.time() - ensemble_start
    p2_time = time.time() - p1_time
    print("=" * 60)
    print(f"Ensemble training complete in {total_time:.1f}s")
    print(f"  Phase 1 (base models): {p1_time - ensemble_start:.1f}s")
    print(f"  Phase 2 (meta-learner): {p2_time:.1f}s")
    print("=" * 60)


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
