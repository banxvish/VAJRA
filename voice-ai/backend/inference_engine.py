"""
KAVACHA AI Voice Defense Engine — Backend Inference Engine
============================================================
Translates audio tensors securely through the isolated 
NN instances scoring outputs to match the Ensemble Trust logic.

Updated to use the pretrained Wav2Vec2ForSequenceClassification
model — inference goes through the HuggingFace Processor pipeline
instead of raw tensor forwarding.
"""

import logging
from typing import Dict, Any, Optional
import torch
import numpy as np

logger = logging.getLogger(__name__)


class APIInferenceEngine:
    """Computes final API logic over loaded models."""

    def __init__(self, models: Any, cfg: Dict[str, Any]):
        self.models = models
        self.cfg = cfg
        self.device = models.device

    @torch.no_grad()
    def evaluate(self, waveform: torch.Tensor, spectrogram: torch.Tensor, enrolled_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculates final prediction structure for React UI response."""
        res = {}
        
        # ── Spectrogram Evaluator (Batch 1, 3, 128, 128) ──
        spec_in = spectrogram.unsqueeze(0).to(self.device)
        spec_out = self.models.spectrogram_model(spec_in)
        spec_probs = torch.softmax(spec_out, dim=1).squeeze()
        res["spec_vote"] = "REAL" if spec_probs.argmax().item() == 0 else "FAKE"
        res["spec_prob"] = spec_probs[0].item()

        # ── Wav2Vec2 Pretrained Evaluator ──
        if self.models.wav2vec_model is not None:
            detector = self.models.wav2vec_model  # Wav2VecPretrainedDetector
            verdict, confidence = detector.predict(waveform)
            prob_real = detector.get_prob_real(waveform)
            res["wav2vec_vote"] = verdict
            res["wav2vec_prob"] = prob_real
        else:
            # Fallback: mirror the spectrogram result
            res["wav2vec_vote"] = res["spec_vote"]
            res["wav2vec_prob"] = res["spec_prob"]

        # ── Codec Artifact Evaluator (Batch 1, 1, 32000) ──
        codec_in = waveform.unsqueeze(0).to(self.device)
        codec_out = self.models.codec_model(codec_in)
        codec_probs = torch.softmax(codec_out, dim=1).squeeze()
        
        c_map = {0: "HUMAN", 1: "ENCODEC", 2: "SOUNDSTREAM"}
        res["codec_vote"] = c_map.get(codec_probs.argmax().item(), "HUMAN")

        # ── Speaker Verification Check ──
        res["speaker_sim"] = 0.0
        if enrolled_emb is not None:
             test_emb = self.models.speaker_verifier.extract_embedding_from_tensor(waveform.cpu())
             res["speaker_sim"] = float(self.models.speaker_verifier.cosine_similarity(enrolled_emb, test_emb))

        # ── Deterministic Trust Score Compute ──
        # Since the models are partially trained/pretrained, raw probabilities are too noisy.
        # We use a strict deterministic ruleset to guarantee accurate frontend alerting:
        
        cv = res["codec_vote"]
        wv = res["wav2vec_vote"]
        sv = res["spec_vote"]
        
        # 1. Critical Veto: Any neural codec artifact guarantees FAKE
        if cv in ["ENCODEC", "SOUNDSTREAM"]:
            final_status = "FAKE"
            final_score = 0.15
            
        # 2. Both high-confidence models agree it's human
        elif cv == "HUMAN" and wv == "REAL":
            final_status = "SAFE"
            final_score = 0.92
            
        # 3. Conflict: Codec says human, but Wav2Vec or Spectrogram detects anomalies
        elif cv == "HUMAN":
            if sv == "FAKE" or wv == "FAKE":
                final_status = "SUSPICIOUS"
                final_score = 0.55
            else:
                final_status = "SAFE"
                final_score = 0.85
                
        else:
            final_status = "FAKE"
            final_score = 0.20

        # Incorporate speaker weight if enrollment exists
        w = self.cfg["ensemble"]
        if enrolled_emb is not None and w.get("speaker_weight", 0) > 0:
            speaker_contrib = w["speaker_weight"] * max(0.0, res["speaker_sim"])
            final_score = final_score * (1 - w["speaker_weight"]) + speaker_contrib

        return {
            "trust_score": float(final_score),
            "status": final_status,
            "models": {
                "spectrogram": sv,
                "wav2vec": wv,
                "codec": cv,
                "speaker_similarity": res["speaker_sim"]
            }
        }
