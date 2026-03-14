"""
KAVACHA AI Voice Defense Engine — Experiment Logger
=====================================================
Saves experiment metadata alongside TensorBoard logs:
  • config snapshot
  • timestamps
  • model architectures
  • final metrics
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Track experiment metadata and results."""

    def __init__(self, log_dir: str = "runs", experiment_name: str = "kavacha"):
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

        self.metadata = {
            "experiment_name": experiment_name,
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "config": {},
            "models": {},
            "metrics": {},
        }

    def log_config(self, cfg: Dict[str, Any]) -> None:
        """Save the training configuration snapshot."""
        self.metadata["config"] = cfg
        self._save()

    def log_model_info(self, model_name: str, info: Dict[str, Any]) -> None:
        """Log model architecture info."""
        self.metadata["models"][model_name] = {
            **info,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def log_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics for a model."""
        clean = {k: v for k, v in metrics.items() if k != "report"}
        self.metadata["metrics"][model_name] = {
            **clean,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def finalize(self, status: str = "completed") -> None:
        """Mark experiment as complete."""
        self.metadata["status"] = status
        self.metadata["completed_at"] = datetime.now().isoformat()
        self._save()
        logger.info(f"Experiment '{self.experiment_name}' → {status}")

    def _save(self) -> None:
        path = os.path.join(self.experiment_dir, "experiment_metadata.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)
