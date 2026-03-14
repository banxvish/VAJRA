"""
KAVACHA AI Voice Defense Engine — Codec Detector Training
==========================================================
Single-stage training for the lightweight 1D CNN codec detector.
"""

import logging
from typing import Any, Dict

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.codec_detector import CodecDetector
from training.trainer import BaseTrainer
from utils.checkpoint import count_parameters

logger = logging.getLogger(__name__)


def train_codec(
    cfg: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "runs",
) -> nn.Module:
    """
    Train the codec artifact detector.
    Returns trained model on CPU.
    """
    c = cfg["codec_training"]

    model = CodecDetector(num_classes=c["num_classes"])
    logger.info(f"Codec detector params: {count_parameters(model, trainable_only=False):,}")

    optimizer = Adam(
        model.parameters(),
        lr=c["lr"],
        weight_decay=c.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=c["epochs"])

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=scheduler,
        gradient_clip_norm=c.get("gradient_clip_norm", 1.0),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        experiment_name="codec",
        logger_name="codec_trainer",
    )

    trainer.fit(
        train_loader, val_loader,
        epochs=c["epochs"],
        stage_name="train",
        patience=c["early_stopping"]["patience"],
    )

    trainer.load_best()
    trainer.close()

    model = model.cpu()
    logger.info(f"Best codec val accuracy: {trainer.best_val_acc:.4f}")
    return model
