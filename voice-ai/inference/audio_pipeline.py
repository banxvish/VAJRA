"""
KAVACHA AI Voice Defense Engine — Deployment Audio Pipeline
=============================================================
Prepares generic incoming audio for all four models identically to 
the training-time preprocessing pipeline.
"""

import logging
from typing import Tuple

import torch
import torchaudio

from preprocessing.audio_processor import load_audio, pad_or_trim, normalize_waveform
from preprocessing.spectrogram_generator import build_mel_transform, generate_spectrogram

logger = logging.getLogger(__name__)


class AudioPipeline:
    """End-to-end inference audio processor."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sr = cfg["audio"]["sample_rate"]
        self.target_len = cfg["audio"]["target_length"]
        self.mel_transform = build_mel_transform(cfg)

    def process_file(self, filepath: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process an audio file completely.
        Returns:
            waveform (1, 32000)
            spectrogram (3, 128, 128)
        """
        waveform, _ = load_audio(filepath, target_sr=self.sr)
        return self.process_tensor(waveform)

    def process_tensor(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process an existing waveform tensor.
        Returns:
            waveform (1, 32000)
            spectrogram (3, 128, 128)
        """
        # Ensure it has a batch dim if completely flattened occasionally
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Mono conversion
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = pad_or_trim(waveform, self.target_len)
        waveform = normalize_waveform(waveform)
        
        # Spectrogram needs shape (1, T) as input inside the generation
        mel = generate_spectrogram(waveform, self.cfg, self.mel_transform)
        
        return waveform, mel
