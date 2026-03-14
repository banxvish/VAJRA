"""
KAVACHA AI Voice Defense Engine — Ensemble Trust Engine
=========================================================
Aggregates logic across the Spectrogram, Wav2Vec2, Codec, and 
Speaker verification models to generate a unified Trust Score.
"""

import logging
from typing import Dict, Any, Tuple
import torch

logger = logging.getLogger(__name__)


class EnsembleEngine:
    """Computes a dynamic deepfake trust score across all algorithms."""
    
    def __init__(self, models, cfg: Dict[str, Any]):
        self.models = models
        self.cfg = cfg
        self.device = models.device

    @torch.no_grad()
    def evaluate(self, waveform: torch.Tensor, spectrogram: torch.Tensor, 
                 enrolled_user_emb=None) -> Dict[str, Any]:
        """
        Run all models efficiently and calculate the final ensemble score.
        Input tensors must be on CPU initially, sent to device directly here.
        """
        results = {}
        
        # Spectrogram Evaluator
        spec_input = spectrogram.unsqueeze(0).to(self.device)  # Add batch dim
        spec_out = self.models.spectrogram_model(spec_input)
        spec_probs = torch.softmax(spec_out, dim=1).squeeze()
        results["spectrogram_vote"] = "REAL" if spec_probs.argmax().item() == 0 else "FAKE"
        results["spectrogram_prob"] = spec_probs[0].item()  # P(REAL)

        # Wav2Vec2 Evaluator
        if self.models.wav2vec_model is not None:
            w2v_input = waveform.to(self.device) # shape (1, 32000)
            w2v_out = self.models.wav2vec_model(w2v_input)
            w2v_probs = torch.softmax(w2v_out, dim=1).squeeze()
            results["wav2vec_vote"] = "REAL" if w2v_probs.argmax().item() == 0 else "FAKE"
            results["wav2vec_prob"] = w2v_probs[0].item()
        else:
            # Fallback to pure spectrogram behavior if Wav2Vec isn't built currently
            results["wav2vec_vote"] = results["spectrogram_vote"]
            results["wav2vec_prob"] = results["spectrogram_prob"]

        # Codec Artifact Evaluator
        codec_input = waveform.unsqueeze(0).to(self.device) # Batch=1, Channel=1
        codec_out = self.models.codec_model(codec_input)
        codec_probs = torch.softmax(codec_out, dim=1).squeeze()
        codec_idx = codec_probs.argmax().item()
        
        c_map = {0: "HUMAN", 1: "ENCODEC", 2: "SOUNDSTREAM"}
        results["codec_vote"] = c_map.get(codec_idx, "UNKNOWN")
        results["codec_prob"] = codec_probs.max().item()

        # Speaker Verification Check
        results["speaker_sim"] = 0.0
        if enrolled_user_emb is not None:
             test_emb = self.models.speaker_verifier.extract_embedding_from_tensor(waveform.cpu())
             sim = self.models.speaker_verifier.cosine_similarity(enrolled_user_emb, test_emb)
             results["speaker_sim"] = sim

        # ---- Ensemble Rules ----
        w = self.cfg["ensemble"]
        codec_score = 1.0 if results["codec_vote"] == "HUMAN" else 0.1
        
        trust_score = (
            w["spectrogram_weight"] * results["spectrogram_prob"] +
            w["wav2vec_weight"] * results["wav2vec_prob"] +
            w["codec_weight"] * codec_score +
            w["speaker_weight"] * max(0.0, results["speaker_sim"])
        )

        results["trust_score"] = float(trust_score)

        th = w["thresholds"]
        if trust_score >= th["safe"]:
            results["status"] = "SAFE"
        elif trust_score >= th["suspicious"]:
            results["status"] = "SUSPICIOUS"
        else:
            results["status"] = "FAKE"

        return results
