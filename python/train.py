import os

from python.ml_pipeline import build_project_paths

TARGET_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3


def main():
    import torch
    from torchvision import datasets, models, transforms

    current_dir = os.getcwd()
    paths = build_project_paths(current_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.CenterCrop(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=paths["train_dir"], transform=transform)
    test_dataset = datasets.ImageFolder(root=paths["test_dir"], transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_dataset.classes)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    if os.path.exists(paths["model_path"]):
        print("Model found, loading...")
        model.load_state_dict(torch.load(paths["model_path"], map_location=device))

    print("Training...")

    for epoch in range(EPOCHS):
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

        avg_train_loss = train_loss / len(train_dataloader)
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_dataloader)

        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Test loss: {avg_test_loss:.4f}")

    print("Done training!")
    print("Saving model...")
    torch.save(model.state_dict(), paths["model_path"])
    print("Done!")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"Test accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    main()
