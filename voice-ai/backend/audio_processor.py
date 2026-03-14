"""
KAVACHA AI Voice Defense Engine — Backend Audio Processor
===========================================================
Handles audio uploaded via API precisely imitating training conditions.
"""

import logging
import os
import tempfile
from typing import Tuple

import torch
import torchaudio

from preprocessing.audio_processor import load_audio, pad_or_trim, normalize_waveform
from preprocessing.spectrogram_generator import build_mel_transform, generate_spectrogram

logger = logging.getLogger(__name__)


class APIAudioProcessor:
    """Preprocesses raw bytes into format suitable for inference."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sr = cfg["audio"]["sample_rate"]
        self.target_len = cfg["audio"]["target_length"]
        self.mel_transform = build_mel_transform(cfg)

    def process_from_bytes(self, audio_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert raw bytes to (waveform, spectrogram) tensors."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            waveform, _ = load_audio(tmp_path, target_sr=self.sr)
            
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
                
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = pad_or_trim(waveform, self.target_len)
            waveform = normalize_waveform(waveform)
            
            mel = generate_spectrogram(waveform, self.cfg, self.mel_transform)
            
            return waveform, mel
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
