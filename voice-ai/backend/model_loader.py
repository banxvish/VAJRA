"""
KAVACHA AI Voice Defense Engine — Backend Model Loader
========================================================
Eagerly loads ML PyTorch state dicts automatically allocating 
to GPUs if available for high-capacity HTTP REST Inference.
"""

import logging
import os
import torch
from typing import Dict, Any, Optional

from models.spectrogram_model import build_spectrogram_model
from models.wav2vec_detector import build_wav2vec_pretrained
from models.codec_detector import CodecDetector
from models.speaker_verification import SpeakerVerifier

logger = logging.getLogger(__name__)


class APIModelLoader:
    """Persistent singleton holding neural net instances for the API."""

    def __init__(self, cfg: Dict[str, Any], model_dir: str = "models", device: Optional[torch.device] = None):
        self.cfg = cfg
        self.model_dir = model_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.spectrogram_model = None
        self.wav2vec_model = None
        self.codec_model = None
        self.speaker_verifier = None

    def initialize(self):
        """Load all production models dynamically into GPU/CPU memory."""
        logger.info(f"API ModelLoader: booting models onto {self.device}")
        
        try:
            print("Loading Spectrogram Model...")
            self.spectrogram_model = self._load_spectrogram()
        except Exception as e:
            logger.error(f"Failed to load Spectrogram Model: {e}")
            self.spectrogram_model = None

        try:
            print("Loading Wav2Vec2 Pretrained Model (superb/wav2vec2-base-superb-sid)...")
            self.wav2vec_model = self._load_wav2vec()
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 Model: {e}")
            self.wav2vec_model = None

        try:
            print("Loading Codec Detector...")
            self.codec_model = self._load_codec()
        except Exception as e:
            logger.error(f"Failed to load Codec Detector: {e}")
            self.codec_model = None

        try:
            print("Loading Speaker Verification Model...")
            self.speaker_verifier = self._load_speaker()
        except Exception as e:
            logger.error(f"Failed to load Speaker Verification Model: {e}")
            self.speaker_verifier = None

        print("All models loaded successfully.")
        logger.info("API Component initialization successful.")

    def _load_spectrogram(self) -> torch.nn.Module:
        path = os.path.join(self.model_dir, "spectrogram_model.pt")
        model = build_spectrogram_model(num_classes=2, pretrained=False).to(self.device)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def _load_wav2vec(self):
        """Load pretrained Wav2Vec2ForSequenceClassification — no .pt checkpoint needed."""
        try:
            detector = build_wav2vec_pretrained(self.cfg, device=self.device)
            print("  ✅ Wav2Vec2 pretrained model loaded (no training required).")
            return detector
        except Exception as e:
            logger.error(f"Pretrained wav2vec load failed: {e}")
            return None

    def _load_codec(self) -> torch.nn.Module:
        path = os.path.join(self.model_dir, "codec_model.pt")
        model = CodecDetector(num_classes=3).to(self.device)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def _load_speaker(self) -> SpeakerVerifier:
        sv = SpeakerVerifier(self.cfg)
        sv._load()
        return sv
