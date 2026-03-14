"""
KAVACHA AI Voice Defense Engine — Evaluation Metrics
=====================================================
Classification metrics + speaker verification EER.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm

from utils.device import get_device


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int = 2,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Compute Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(inputs)

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds)
        all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except ValueError:
        roc_auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate from genuine/impostor similarity scores."""
    thresholds = np.sort(np.unique(np.concatenate([genuine_scores, impostor_scores])))
    far = np.array([np.mean(impostor_scores >= t) for t in thresholds])
    frr = np.array([np.mean(genuine_scores < t) for t in thresholds])
    idx = np.argmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2.0
    return eer, thresholds[idx]
