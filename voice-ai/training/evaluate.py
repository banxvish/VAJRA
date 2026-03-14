"""
KAVACHA Voice Deepfake Detection — Evaluation Module
=====================================================
Comprehensive evaluation metrics for:
  1. Classification models (spectrogram & codec detectors)
     • Accuracy, Precision, Recall, F1, ROC-AUC
  2. Speaker verification
     • Equal Error Rate (EER)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from training.utils import get_device, setup_logger

logger = setup_logger("evaluator")


# ------------------------------------------------------------------
# Classification Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate a classification model on the given dataloader.

    Returns
    -------
    metrics : dict with keys
        accuracy, precision, recall, f1, roc_auc
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

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

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (binary vs multi-class)
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )
    except ValueError:
        roc_auc = 0.0
        logger.warning("ROC-AUC could not be computed (insufficient classes in batch).")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    # Pretty print
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k:>12s}: {v:.4f}")
    logger.info("-" * 50)
    logger.info("\nClassification Report:")
    logger.info(
        classification_report(
            y_true, y_pred, digits=4, zero_division=0
        )
    )

    return metrics


# ------------------------------------------------------------------
# Speaker Verification — Equal Error Rate
# ------------------------------------------------------------------

def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) from genuine and impostor similarity scores.

    Parameters
    ----------
    genuine_scores : np.ndarray
        Cosine similarities for genuine (same speaker) pairs.
    impostor_scores : np.ndarray
        Cosine similarities for impostor (different speaker) pairs.

    Returns
    -------
    eer : float
        Equal Error Rate (0–1)
    threshold : float
        Optimal threshold at EER point.
    """
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])

    # Sort thresholds
    thresholds = np.sort(np.unique(scores))

    far_list = []
    frr_list = []

    for thresh in thresholds:
        # False Acceptance Rate: impostor accepted
        far = np.mean(impostor_scores >= thresh)
        # False Rejection Rate: genuine rejected
        frr = np.mean(genuine_scores < thresh)
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # Find crossover point
    abs_diff = np.abs(far_arr - frr_arr)
    idx = np.argmin(abs_diff)

    eer = (far_arr[idx] + frr_arr[idx]) / 2.0
    threshold = thresholds[idx]

    logger.info(f"EER: {eer:.4f} at threshold: {threshold:.4f}")
    return eer, threshold


def evaluate_speaker_verification(
    verifier,
    genuine_pairs: List[Tuple[str, str]],
    impostor_pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    """
    Evaluate speaker verification by computing EER from genuine/impostor pairs.

    Parameters
    ----------
    verifier : SpeakerVerifier
        Initialized speaker verification wrapper.
    genuine_pairs : list of (audio_path_1, audio_path_2) — same speaker
    impostor_pairs : list of (audio_path_1, audio_path_2) — different speakers

    Returns
    -------
    metrics : dict with eer, threshold
    """
    genuine_scores = []
    for path_a, path_b in tqdm(genuine_pairs, desc="Genuine pairs"):
        emb_a = verifier.extract_embedding(path_a)
        emb_b = verifier.extract_embedding(path_b)
        sim = verifier._cosine_similarity(emb_a, emb_b)
        genuine_scores.append(sim)

    impostor_scores = []
    for path_a, path_b in tqdm(impostor_pairs, desc="Impostor pairs"):
        emb_a = verifier.extract_embedding(path_a)
        emb_b = verifier.extract_embedding(path_b)
        sim = verifier._cosine_similarity(emb_a, emb_b)
        impostor_scores.append(sim)

    eer, threshold = compute_eer(
        np.array(genuine_scores), np.array(impostor_scores)
    )

    return {"eer": eer, "threshold": threshold}
