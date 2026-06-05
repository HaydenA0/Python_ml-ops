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

        self.assertEqual(paths["data_dir"], "/workspace/project/data")
        self.assertEqual(paths["images_dir"], "/workspace/project/data/Training/Images")
        self.assertEqual(paths["train_metadata_csv"], "/workspace/project/data/stage2_train_metadata.csv")
        self.assertEqual(paths["model_path"], "/workspace/project/models/model.pth")


class CountImagesByLabelTests(unittest.TestCase):
    def test_count_images_by_label_ignores_non_directories_and_sums_total(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            normal_dir = test_dir / "NORMAL"
            pneumonia_dir = test_dir / "PNEUMONIA"
            normal_dir.mkdir()
            pneumonia_dir.mkdir()

            (normal_dir / "img1.jpeg").write_text("x", encoding="utf-8")
            (normal_dir / "img2.jpeg").write_text("x", encoding="utf-8")
            (pneumonia_dir / "img3.jpeg").write_text("x", encoding="utf-8")
            (test_dir / "README.txt").write_text("ignore me", encoding="utf-8")

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


try:
    from python.pcxp_mlops.model import BaseModelWrapper, StackingEnsemble, build_tiny_cnn
except ModuleNotFoundError:
    BaseModelWrapper = None
    StackingEnsemble = None
    build_tiny_cnn = None


@unittest.skipIf(StackingEnsemble is None, "torch is not installed")
class BaseModelWrapperTests(unittest.TestCase):
    def test_wrapper_stores_name_and_model(self):
        import torch
        model = torch.nn.Linear(2, 2)
        wrapper = BaseModelWrapper(model, "test-model", "cpu")
        self.assertEqual(wrapper.name, "test-model")
        self.assertIs(wrapper.model, model)


@unittest.skipIf(StackingEnsemble is None, "torch is not installed")
class StackingEnsembleTests(unittest.TestCase):
    def _fixed_model(self, val):
        import torch
        class _Fixed(torch.nn.Module):
            def __init__(self, v):
                super().__init__()
                self.v = v
            def forward(self, x):
                b = x.size(0)
                logits = torch.zeros(b, 2)
                logits[:, 1] = self.v
                return logits
        return _Fixed(val)

    def test_ensemble_averages_predictions_when_no_meta_model_or_weights(self):
        import torch
        vals = [0.8, 0.6, 0.4]
        wrappers = [BaseModelWrapper(self._fixed_model(v), str(v), "cpu") for v in vals]
        ensemble = StackingEnsemble(wrappers)
        x = torch.randn(2, 3, 224, 224)
        result = ensemble.predict(x, run_parallel=False)
        expected = sum(torch.sigmoid(torch.tensor(v)).item() for v in vals) / len(vals)
        self.assertAlmostEqual(result[0], expected, places=5)
        self.assertAlmostEqual(result[1], expected, places=5)

    def test_ensemble_uses_heuristic_weights(self):
        import torch
        vals = [0.9, 0.5]
        weights = [0.75, 0.25]
        wrappers = [BaseModelWrapper(self._fixed_model(v), str(v), "cpu") for v in vals]
        ensemble = StackingEnsemble(wrappers, weights=weights)
        x = torch.randn(1, 3, 224, 224)
        result = ensemble.predict(x, run_parallel=False)
        sigmoids = [torch.sigmoid(torch.tensor(v)).item() for v in vals]
        expected = sum(w * s for w, s in zip(weights, sigmoids)) / sum(weights)
        self.assertAlmostEqual(result[0], expected, places=5)

    def test_ensemble_returns_list(self):
        import torch
        wrapper = BaseModelWrapper(self._fixed_model(0.8), "single", "cpu")
        ensemble = StackingEnsemble([wrapper])
        x = torch.randn(3, 3, 224, 224)
        result = ensemble.predict(x, run_parallel=False)
        expected = torch.sigmoid(torch.tensor(0.8)).item()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], expected, places=5)


@unittest.skipIf(build_tiny_cnn is None, "torch is not installed")
class BuildTinyCnnTests(unittest.TestCase):
    def test_tiny_cnn_output_shape(self):
        import torch
        model = build_tiny_cnn(2)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 2))
