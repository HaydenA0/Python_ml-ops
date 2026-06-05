"""Model factory functions."""
from __future__ import annotations


def build_resnet18(num_classes: int, pretrained: bool = True):
    import torch
    from torchvision import models
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def build_tiny_cnn(num_classes: int):
    import torch
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(128, num_classes),
    )


class BaseModelWrapper:
    def __init__(self, model, name: str, device):
        self.model = model.to(device).eval()
        self.name = name

    def predict(self, x) -> list[float]:
        import torch
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        return probs[:, 1].cpu().tolist()


class StackingEnsemble:
    def __init__(self, base_models: list[BaseModelWrapper], meta_model=None, weights: list[float] | None = None):
        self.base_models = base_models
        self.meta_model = meta_model
        self._weights = weights

    def _get_base_predictions_sequential(self, x) -> list[list[float]]:
        return [m.predict(x) for m in self.base_models]

    def _get_base_predictions_parallel(self, x) -> list[list[float]]:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.base_models)) as executor:
            indexed = {executor.submit(m.predict, x): i for i, m in enumerate(self.base_models)}
            results: list[list[float]] = [[] for _ in self.base_models]
            for f in concurrent.futures.as_completed(indexed):
                results[indexed[f]] = f.result()
        return results

    def get_base_predictions(self, x, run_parallel: bool = True) -> dict[str, list[float]]:
        if run_parallel and len(self.base_models) > 1:
            base_preds = self._get_base_predictions_parallel(x)
        else:
            base_preds = self._get_base_predictions_sequential(x)
        return {m.name: preds for m, preds in zip(self.base_models, base_preds)}

    def predict(self, x, run_parallel: bool = True, metadata=None) -> list[float]:
        import numpy as np
        if run_parallel and len(self.base_models) > 1:
            base_preds = self._get_base_predictions_parallel(x)
        else:
            base_preds = self._get_base_predictions_sequential(x)

        batch_size = len(base_preds[0])

        if self.meta_model is not None:
            meta_input = np.column_stack(base_preds)
            if metadata is not None:
                meta_input = np.column_stack([meta_input, metadata])
            if hasattr(self.meta_model, "predict_proba"):
                final = self.meta_model.predict_proba(meta_input)[:, 1].tolist()
            else:
                final = self.meta_model.predict(meta_input).tolist()
        elif self._weights is not None:
            total = sum(self._weights) or 1.0
            final = [
                sum(w * base_preds[m][i] for m, w in enumerate(self._weights)) / total
                for i in range(batch_size)
            ]
        else:
            final = [
                sum(base_preds[m][i] for m in range(len(self.base_models))) / len(self.base_models)
                for i in range(batch_size)
            ]

        return final
