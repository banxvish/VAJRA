"""
KAVACHA AI Voice Defense Engine — Deployment Model Loader
===========================================================
Loads pre-trained models into evaluation mode. Supports automatic
GPU / CPU device placement for real-time inference.
"""

import os
import logging
from typing import Dict, Any, Optional

import torch

from models.spectrogram_model import build_spectrogram_model
from models.wav2vec_detector import build_wav2vec_model
from models.codec_detector import CodecDetector
from models.speaker_verification import SpeakerVerifier

logger = logging.getLogger(__name__)


class ModelLoader:
    """Manages the loading and memory allocation of all KAVACHA models."""

    def __init__(self, cfg: Dict[str, Any], model_dir: str = "models", device: Optional[torch.device] = None):
        self.cfg = cfg
        self.model_dir = model_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.spectrogram_model = None
        self.wav2vec_model = None
        self.codec_model = None
        self.speaker_verifier = None

    def load_all(self):
        """Load all four models into the specified device in eval mode."""
        logger.info(f"Loading KAVACHA models onto {self.device}...")
        self.spectrogram_model = self._load_spectrogram()
        self.wav2vec_model = self._load_wav2vec()
        self.codec_model = self._load_codec()
        self.speaker_verifier = self._load_speaker_verifier()
        logger.info("All models successfully loaded in eval mode.")

    def _load_spectrogram(self) -> torch.nn.Module:
        path = os.path.join(self.model_dir, "spectrogram_model.pt")
        model = build_spectrogram_model(num_classes=2, pretrained=False).to(self.device)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            logger.warning(f"Spectrogram weights not found at {path}. Using untrained weights.")
        model.eval()
        return model

    def _load_wav2vec(self) -> torch.nn.Module:
        path = os.path.join(self.model_dir, "wav2vec_detector.pt")
        try:
            model = build_wav2vec_model(self.cfg).to(self.device)
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                logger.warning(f"Wav2Vec weights not found at {path}. Using untrained head.")
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {e}")
            return None

    def _load_codec(self) -> torch.nn.Module:
        path = os.path.join(self.model_dir, "codec_model.pt")
        model = CodecDetector(num_classes=3).to(self.device)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            logger.warning(f"Codec weights not found at {path}. Using untrained weights.")
        model.eval()
        return model

    def _load_speaker_verifier(self) -> SpeakerVerifier:
        sv = SpeakerVerifier(self.cfg)
        sv._load() # Preload the ECAPA-TDNN graph eagerly
        return sv
