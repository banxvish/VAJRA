"""
KAVACHA AI Voice Defense Engine — Base Trainer
================================================
Reusable training loop with:
  • Mixed-precision (AMP)
  • Gradient clipping
  • TensorBoard logging
  • Checkpoint management
  • Early stopping
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.checkpoint import EarlyStopping, save_checkpoint, ensure_dirs, setup_logger
from utils.device import get_device


class BaseTrainer:
    """
    Generic training/validation loop with AMP, gradient clipping,
    TensorBoard, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler=None,
        device: Optional[torch.device] = None,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs",
        experiment_name: str = "model",
        logger_name: str = "trainer",
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.clip_norm = gradient_clip_norm
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.logger = setup_logger(logger_name)

        ensure_dirs(checkpoint_dir, log_dir)
        self.writer = SummaryWriter(
            log_dir=os.path.join(log_dir, experiment_name)
        )
        self.best_val_acc = 0.0
        self.global_epoch = 0

    def train_one_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """One training epoch with AMP + gradient clipping."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(dataloader, desc="  Train", leave=False):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type,
                          enabled=(self.device.type == "cuda")):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validation epoch."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(dataloader, desc="  Val  ", leave=False):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type,
                          enabled=(self.device.type == "cuda")):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        stage_name: str = "train",
        patience: int = 5,
    ) -> None:
        """Run training loop for N epochs with early stopping."""
        early_stop = EarlyStopping(patience=patience, mode="min")
        prefix = f"{self.experiment_name}/{stage_name}"

        for epoch in range(1, epochs + 1):
            self.global_epoch += 1
            self.logger.info(f"{stage_name} — Epoch {epoch}/{epochs}")

            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"  Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | LR={lr:.2e}"
            )

            # TensorBoard
            self.writer.add_scalar(f"{prefix}/train_loss", train_loss, self.global_epoch)
            self.writer.add_scalar(f"{prefix}/val_loss", val_loss, self.global_epoch)
            self.writer.add_scalar(f"{prefix}/train_acc", train_acc, self.global_epoch)
            self.writer.add_scalar(f"{prefix}/val_acc", val_acc, self.global_epoch)
            self.writer.add_scalar(f"{prefix}/lr", lr, self.global_epoch)

            # Best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_checkpoint(
                    self.model, self.optimizer, self.global_epoch, val_acc,
                    os.path.join(self.checkpoint_dir,
                                 f"{self.experiment_name}_best.pt"),
                )
                self.logger.info(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

            # Last checkpoint
            save_checkpoint(
                self.model, self.optimizer, self.global_epoch, val_acc,
                os.path.join(self.checkpoint_dir,
                             f"{self.experiment_name}_last.pt"),
            )

            # Early stopping
            if early_stop(val_loss):
                self.logger.info(f"  Early stopping at epoch {epoch}.")
                break

    def load_best(self) -> None:
        """Load best checkpoint into model."""
        path = os.path.join(self.checkpoint_dir,
                            f"{self.experiment_name}_best.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.logger.info(f"Loaded best checkpoint: {path}")

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()
