"""
KAVACHA AI Voice Defense Engine — Checkpoint Management
========================================================
Save/load training checkpoints and early stopping logic.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Checkpoint I/O
# ------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model + optimizer state with metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load model + optimizer state from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ------------------------------------------------------------------
# Early Stopping
# ------------------------------------------------------------------

class EarlyStopping:
    """Stop training when monitored metric does not improve."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best: Optional[float] = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False

        improved = (
            (metric < self.best - self.min_delta) if self.mode == "min"
            else (metric > self.best + self.min_delta)
        )

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False

    def reset(self) -> None:
        self.counter = 0
        self.best = None
        self.should_stop = False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_config(config_path: str = "configs/training.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console + optional file handlers."""
    lgr = logging.getLogger(name)
    lgr.setLevel(level)
    lgr.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    lgr.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        lgr.addHandler(fh)

    return lgr
