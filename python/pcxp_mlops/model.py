"""Model factory functions."""

from __future__ import annotations


def build_resnet18(num_classes: int, pretrained: bool = True):
    """Build the project classifier architecture."""
    import torch
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
