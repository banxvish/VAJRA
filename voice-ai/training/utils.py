"""
KAVACHA Voice Deepfake Detection — Utility Helpers
===================================================
Shared utility functions used across the training pipeline:
  • Configuration loading
  • Seed initialization
  • Device selection
  • Logging setup
  • Checkpoint management
  • Early stopping
"""

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

def load_config(config_path: str = "configs/training.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def get_device() -> torch.device:
    """Select GPU if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def setup_logger(name: str, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint."""
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
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ------------------------------------------------------------------
# Early Stopping
# ------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == "min":
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


# ------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------

def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
