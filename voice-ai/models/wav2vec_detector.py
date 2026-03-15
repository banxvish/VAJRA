"""
KAVACHA AI Voice Defense Engine — Wav2Vec2 Pretrained Deepfake Detector
=========================================================================
Zero-training deepfake detection using a pretrained HuggingFace
Wav2Vec2ForSequenceClassification model (superb/wav2vec2-base-superb-sid).

This model leverages fully pretrained weights — no custom training required.
It produces classification logits directly, which are softmaxed to obtain
per-class confidence probabilities for the SAFE/FAKE verdict.

Architecture:
  Wav2Vec2 feature extractor + transformer encoder (pretrained)
  → Built-in classification head (pretrained)
  → Softmax → argmax for verdict
"""

import os
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


class Wav2VecPretrainedDetector:
    """
    Pretrained Wav2Vec2 deepfake classifier wrapper.

    Uses `superb/wav2vec2-base-superb-sid` which ships with a fully
    trained classification head — no fine-tuning needed.

    Input:  raw waveform (1-D numpy or tensor) at 16 kHz
    Output: (verdict_str, confidence_float)
    """

    def __init__(
        self,
        model_name: str = "superb/wav2vec2-base-superb-sid",
        device: torch.device = None,
    ):
        self.model_name = model_name
        self.device = device or torch.device("cpu")

        # ── Integration with new Training Pipeline ──
        # Check if we have fine-tuned local weights. If yes, override HF download.
        local_trained_path = "export/wav2vec_finetuned_hf"
        if os.path.exists(local_trained_path):
            print(f"Loading FINE-TUNED Wav2Vec2 weights from {local_trained_path}")
            load_path = local_trained_path
        else:
            load_path = model_name

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(load_path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(load_path)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, waveform: torch.Tensor) -> Tuple[str, float]:
        """
        Run inference on a single waveform tensor.

        Parameters
        ----------
        waveform : Tensor of shape (1, samples) or (samples,)
            Raw waveform at 16 kHz.

        Returns
        -------
        verdict : str — "REAL" or "FAKE"
        confidence : float — softmax probability for the winning class
        """
        # Flatten to 1-D float for the processor
        wav_np = waveform.squeeze().cpu().numpy()

        inputs = self.processor(
            wav_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze()

        prediction = torch.argmax(probs).item()
        confidence = torch.max(probs).item()

        # Class 0 → REAL / SAFE, any other class → FAKE
        verdict = "REAL" if prediction == 0 else "FAKE"
        return verdict, confidence

    def get_prob_real(self, waveform: torch.Tensor) -> float:
        """Return the *real* class probability (class 0) for ensemble scoring."""
        wav_np = waveform.squeeze().cpu().numpy()
        inputs = self.processor(
            wav_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze()

        return probs[0].item()


def build_wav2vec_pretrained(cfg: Dict[str, Any], device: torch.device = None) -> Wav2VecPretrainedDetector:
    """Build the pretrained Wav2Vec2 detector from config."""
    model_name = cfg["wav2vec"].get("model_name", "superb/wav2vec2-base-superb-sid")
    return Wav2VecPretrainedDetector(model_name=model_name, device=device)


# ---- legacy alias kept for backward-compat imports ----
def build_wav2vec_model(cfg: Dict[str, Any], device: torch.device = None) -> Wav2VecPretrainedDetector:
    """Alias forwarding to build_wav2vec_pretrained."""
    return build_wav2vec_pretrained(cfg, device=device)
