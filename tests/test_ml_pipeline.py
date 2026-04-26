import tempfile
import unittest
from pathlib import Path

from python.ml_pipeline import (
    apply_threshold,
    build_project_paths,
    count_images_by_label,
    recall_from_confusion_matrix,
)


class BuildProjectPathsTests(unittest.TestCase):
    def test_build_project_paths_uses_expected_layout(self):
        paths = build_project_paths("/workspace/project")

        self.assertEqual(paths["data_dir"], "/workspace/project/data/PCXP")
        self.assertEqual(paths["train_dir"], "/workspace/project/data/PCXP/train")
        self.assertEqual(paths["test_dir"], "/workspace/project/data/PCXP/test")
        self.assertEqual(paths["model_path"], "/workspace/project/models/model.pth")


class CountImagesByLabelTests(unittest.TestCase):
    def test_count_images_by_label_ignores_non_directories_and_sums_total(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            normal_dir = test_dir / "NORMAL"
            pneumonia_dir = test_dir / "PNEUMONIA"
            normal_dir.mkdir()
            pneumonia_dir.mkdir()

            (normal_dir / "img1.jpeg").write_text("x")
            (normal_dir / "img2.jpeg").write_text("x")
            (pneumonia_dir / "img3.jpeg").write_text("x")
            (test_dir / "README.txt").write_text("ignore me")

            counts = count_images_by_label(str(test_dir))

        self.assertEqual(counts, [("NORMAL", 2), ("PNEUMONIA", 1), ("Total", 3)])


class ApplyThresholdTests(unittest.TestCase):
    def test_apply_threshold_marks_positive_class_at_or_above_threshold(self):
        probabilities = [
            [0.20, 0.80],
            [0.13, 0.87],
            [0.90, 0.10],
        ]

        preds = apply_threshold(probabilities, 0.87)

        self.assertEqual(preds, [0, 1, 0])


class RecallFromConfusionMatrixTests(unittest.TestCase):
    def test_recall_from_confusion_matrix_returns_positive_class_recall(self):
        confusion = [
            [8, 1],
            [2, 6],
        ]

        recall = recall_from_confusion_matrix(confusion)

        self.assertEqual(recall, 0.75)

    def test_recall_from_confusion_matrix_handles_no_positive_examples(self):
        confusion = [
            [5, 0],
            [0, 0],
        ]

        recall = recall_from_confusion_matrix(confusion)

        self.assertEqual(recall, 0.0)


if __name__ == "__main__":
    unittest.main()
