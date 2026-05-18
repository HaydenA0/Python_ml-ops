"""Training pipeline for the PCXP classifier."""

from __future__ import annotations

import json

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TARGET_SIZE,
    get_paths,
)
from .data_loader import load_datasets
from .metadata import default_metadata, save_metadata, stamp_training_time
from .model import build_resnet18


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

    print("Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_dataloader), 1)
        avg_test_loss = test_loss / max(len(test_dataloader), 1)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Test loss: {avg_test_loss:.4f}"
        )

    torch.save(model.state_dict(), paths.model_path)

    with paths.classes_path.open("w", encoding="utf-8") as handle:
        json.dump(train_dataset.classes, handle, indent=2)

    metadata = default_metadata()
    metadata["class_names"] = train_dataset.classes
    metadata["model_version"] = DEFAULT_MODEL_VERSION
    metadata = stamp_training_time(metadata)
    save_metadata(paths, metadata)
    print(f"Saved model to {paths.model_path}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
