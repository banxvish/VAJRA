"""
KAVACHA AI Voice Defense Engine — Spectrogram Deepfake Detector
================================================================
EfficientNet-B0 CNN pretrained on ImageNet for binary classification
of mel-spectrogram images.

Custom classifier head:
  Dropout → Linear(1280, 512) → ReLU → Linear(512, 2)
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision import models


def build_spectrogram_model(
    num_classes: int = 2,
    dropout: float = 0.3,
    hidden_dim: int = 512,
    pretrained: bool = True,
) -> nn.Module:
    """Load pretrained EfficientNet-B0 with custom classifier head."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features  # 1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 2) -> None:
    """Unfreeze last N feature blocks + classifier."""
    for param in model.classifier.parameters():
        param.requires_grad = True
    total = len(model.features)
    start = max(0, total - num_blocks)
    for i, block in enumerate(model.features):
        for param in block.parameters():
            param.requires_grad = (i >= start)
