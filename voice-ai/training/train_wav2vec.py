"""
KAVACHA AI Voice Defense Engine — Wav2Vec2 Detector Training
==============================================================
Two-stage transfer learning for the self-supervised deepfake detector.

Stage 1: Freeze Wav2Vec2 transformer → train classifier head
Stage 2: Unfreeze top N transformer layers → fine-tune
"""

import logging
from typing import Any, Dict

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.wav2vec_detector import (
    build_wav2vec_model,
    freeze_wav2vec_encoder,
    unfreeze_wav2vec_top_layers,
)
from training.trainer import BaseTrainer
from utils.checkpoint import count_parameters

logger = logging.getLogger(__name__)


def train_wav2vec(
    cfg: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "runs",
) -> nn.Module:
    """
    Full two-stage Wav2Vec2 detector training.
    Returns trained model on CPU.
    """
    w = cfg["wav2vec"]

    # Build model
    model = build_wav2vec_model(cfg)

    # ---- STAGE 1: Classifier head only ----
    logger.info("=" * 60)
    logger.info("WAV2VEC2 — Stage 1: Classifier head training")
    logger.info("=" * 60)

    freeze_wav2vec_encoder(model)
    logger.info(f"Trainable params (stage 1): {count_parameters(model):,}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=w["lr_stage1"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=w["epochs_stage1"])

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=scheduler,
        gradient_clip_norm=w.get("gradient_clip_norm", 1.0),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        experiment_name="wav2vec",
        logger_name="wav2vec_trainer",
    )

    trainer.fit(
        train_loader, val_loader,
        epochs=w["epochs_stage1"],
        stage_name="stage1",
        patience=w["early_stopping"]["patience"],
    )

    # ---- STAGE 2: Fine-tune top transformer layers ----
    logger.info("=" * 60)
    logger.info("WAV2VEC2 — Stage 2: Fine-tuning top transformer layers")
    logger.info("=" * 60)

    unfreeze_wav2vec_top_layers(model, num_layers=w["unfreeze_last_n_layers"])
    logger.info(f"Trainable params (stage 2): {count_parameters(model):,}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=w["lr_stage2"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=w["epochs_stage2"])

    trainer.optimizer = optimizer
    trainer.scheduler = scheduler

    trainer.fit(
        train_loader, val_loader,
        epochs=w["epochs_stage2"],
        stage_name="stage2",
        patience=w["early_stopping"]["patience"],
    )

    trainer.load_best()
    trainer.close()

    model = model.cpu()
    logger.info(f"Best wav2vec val accuracy: {trainer.best_val_acc:.4f}")
    return model
