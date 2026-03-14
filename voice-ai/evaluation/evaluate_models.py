"""
KAVACHA AI Voice Defense Engine — Model Evaluator
===================================================
Orchestrates evaluation of all trained models and prints results.
"""

import logging
from typing import Any, Dict, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import evaluate_classifier

logger = logging.getLogger(__name__)


def evaluate_all_models(
    spectrogram_model: Optional[nn.Module],
    wav2vec_model: Optional[nn.Module],
    codec_model: Optional[nn.Module],
    spec_loader: Optional[DataLoader],
    wav2vec_loader: Optional[DataLoader],
    codec_loader: Optional[DataLoader],
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all available models and return metrics dict.
    """
    results = {}

    if spectrogram_model and spec_loader:
        logger.info("=" * 50)
        logger.info("EVALUATING: Spectrogram Deepfake Detector")
        logger.info("=" * 50)
        m = evaluate_classifier(
            spectrogram_model, spec_loader,
            num_classes=cfg["training"]["num_classes"],
        )
        results["spectrogram"] = m
        _print_metrics("Spectrogram", m)

    if wav2vec_model and wav2vec_loader:
        logger.info("=" * 50)
        logger.info("EVALUATING: Wav2Vec2 Deepfake Detector")
        logger.info("=" * 50)
        m = evaluate_classifier(
            wav2vec_model, wav2vec_loader,
            num_classes=cfg["wav2vec"]["num_classes"],
        )
        results["wav2vec"] = m
        _print_metrics("Wav2Vec2", m)

    if codec_model and codec_loader:
        logger.info("=" * 50)
        logger.info("EVALUATING: Codec Artifact Detector")
        logger.info("=" * 50)
        m = evaluate_classifier(
            codec_model, codec_loader,
            num_classes=cfg["codec_training"]["num_classes"],
        )
        results["codec"] = m
        _print_metrics("Codec", m)

    return results


def _print_metrics(name: str, metrics: Dict[str, float]) -> None:
    logger.info(f"\n{'─' * 40}")
    logger.info(f"  {name} Results:")
    logger.info(f"{'─' * 40}")
    for k, v in metrics.items():
        if k == "report":
            logger.info(f"\n{v}")
        else:
            logger.info(f"  {k:>12s}: {v:.4f}")
