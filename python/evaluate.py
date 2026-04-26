import os

from python.ml_pipeline import (
    apply_threshold,
    build_project_paths,
    count_images_by_label,
    recall_from_confusion_matrix,
)

TARGET_SIZE = 224
BATCH_SIZE = 16
THRESHOLD = 0.87


def main():
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import confusion_matrix
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
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_dataset.classes)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    if os.path.exists(paths["model_path"]):
        print("Model found, loading...")
        model.load_state_dict(torch.load(paths["model_path"], map_location=device))

    print(count_images_by_label(paths["test_dir"]))

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().tolist()
            preds = apply_threshold(probs, THRESHOLD)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    recall = recall_from_confusion_matrix(cm)
    print(f"Using threshold: {THRESHOLD}")
    print("Confusion matrix:")
    print(cm)
    print(f"Pneumonia recall: {recall:.4f}")


if __name__ == "__main__":
    main()
