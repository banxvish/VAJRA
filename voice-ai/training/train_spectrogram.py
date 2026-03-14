"""
KAVACHA AI Voice Defense Engine — Spectrogram Model Training
==============================================================
Two-stage EfficientNet-B0 transfer learning pipeline.

Stage 1: Freeze backbone → train classifier head
Stage 2: Unfreeze last blocks → fine-tune full model
"""

import logging
from typing import Any, Dict

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.spectrogram_model import (
    build_spectrogram_model, freeze_backbone, unfreeze_last_blocks,
)
from training.trainer import BaseTrainer
from utils.checkpoint import count_parameters

logger = logging.getLogger(__name__)


def train_spectrogram(
    cfg: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "runs",
) -> nn.Module:
    """
    Full two-stage spectrogram model training.
    Returns trained model on CPU.
    """
    t = cfg["training"]

    # Build model
    model = build_spectrogram_model(
        num_classes=t["num_classes"],
        dropout=t["dropout"],
        hidden_dim=t["hidden_dim"],
        pretrained=True,
    )

    # ---- STAGE 1: Classifier head only ----
    logger.info("=" * 60)
    logger.info("SPECTROGRAM — Stage 1: Classifier head training")
    logger.info("=" * 60)

    freeze_backbone(model)
    logger.info(f"Trainable params (stage 1): {count_parameters(model):,}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t["optimizer"]["lr_stage1"],
        weight_decay=t["optimizer"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t["scheduler"]["T_max_stage1"])

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=scheduler,
        gradient_clip_norm=t.get("gradient_clip_norm", 1.0),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        experiment_name="spectrogram",
        logger_name="spec_trainer",
    )

    trainer.fit(
        train_loader, val_loader,
        epochs=t["epochs_stage1"],
        stage_name="stage1",
        patience=t["early_stopping"]["patience"],
    )

    # ---- STAGE 2: Fine-tune backbone ----
    logger.info("=" * 60)
    logger.info("SPECTROGRAM — Stage 2: Fine-tuning backbone")
    logger.info("=" * 60)

    unfreeze_last_blocks(model, num_blocks=2)
    logger.info(f"Trainable params (stage 2): {count_parameters(model):,}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t["optimizer"]["lr_stage2"],
        weight_decay=t["optimizer"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t["scheduler"]["T_max_stage2"])

    trainer.optimizer = optimizer
    trainer.scheduler = scheduler

    trainer.fit(
        train_loader, val_loader,
        epochs=t["epochs_stage2"],
        stage_name="stage2",
        patience=t["early_stopping"]["patience"],
    )

    trainer.load_best()
    trainer.close()

    model = model.cpu()
    logger.info(f"Best spectrogram val accuracy: {trainer.best_val_acc:.4f}")
    return model
