import os


def build_project_paths(current_dir):
    data_dir = os.path.join(current_dir, "data", "PCXP")
    return {
        "current_dir": current_dir,
        "data_dir": data_dir,
        "train_dir": os.path.join(data_dir, "train"),
        "test_dir": os.path.join(data_dir, "test"),
        "model_dir": os.path.join(current_dir, "models"),
        "model_path": os.path.join(current_dir, "models", "model.pth"),
    }


def count_images_by_label(test_dir):
    counts = []
    total = 0

    for label in sorted(os.listdir(test_dir)):
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue

        image_count = sum(
            1 for entry in os.listdir(label_dir)
            if os.path.isfile(os.path.join(label_dir, entry))
        )
        counts.append((label, image_count))
        total += image_count

    counts.append(("Total", total))
    return counts


def apply_threshold(probabilities, threshold, positive_index=1):
    return [
        1 if class_probs[positive_index] >= threshold else 0
        for class_probs in probabilities
    ]


def recall_from_confusion_matrix(confusion):
    true_positives = confusion[1][1]
    false_negatives = confusion[1][0]
    denominator = false_negatives + true_positives
    if denominator == 0:
        return 0.0
    return true_positives / denominator
