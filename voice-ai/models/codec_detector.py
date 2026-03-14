"""
KAVACHA AI Voice Defense Engine — Codec Artifact Detector
==========================================================
Lightweight 1D CNN for detecting neural codec artifacts.

Architecture:
  Conv1d(1, 64, k=7, s=2) → BN → ReLU
  Conv1d(64, 128, k=5, s=2) → BN → ReLU
  Conv1d(128, 256, k=3, s=2) → BN → ReLU
  AdaptiveAvgPool1d(1)
  Linear(256, 3)

Classes: HUMAN (0) | ENCODEC (1) | SOUNDSTREAM (2)
"""

import torch
import torch.nn as nn


class CodecDetector(nn.Module):
    """
    1D CNN for codec artifact classification.

    Input:  (batch, 1, 32000)
    Output: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
