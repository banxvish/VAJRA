"""
KAVACHA AI Voice Defense Engine — Audio Processor
===================================================
Core audio preprocessing pipeline:
  1. Load audio file
  2. Resample to 16 kHz
  3. Convert to mono
  4. Trim or pad to exactly 2 seconds (32 000 samples)
  5. Normalize waveform (zero-mean, unit-variance)
"""

from typing import Any, Dict, Tuple

import torch
import torchaudio
import torchaudio.transforms as T


def load_audio(path: str, target_sr: int = 16_000) -> Tuple[torch.Tensor, int]:
    """
    Load audio file → resample → mono.

    Returns
    -------
    waveform : Tensor (1, num_samples)
    sample_rate : int
    """
    import soundfile as sf
    import numpy as np
    
    try:
        data, sr = sf.read(path)
    except Exception:
        # Fallback to librosa which supports mp3 via built-in packages
        import librosa
        data, sr = librosa.load(path, sr=None, mono=False)
        # librosa returns shape (channels, length), make it (length, channels)
        if data.ndim == 2:
            data = data.T
            
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    waveform = torch.tensor(data.astype(np.float32)).T

    # Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        waveform = T.Resample(sr, target_sr)(waveform)

    return waveform, target_sr


def pad_or_trim(waveform: torch.Tensor, target_length: int = 32_000) -> torch.Tensor:
    """Pad (zeros) or trim waveform to exactly *target_length* samples."""
    n = waveform.shape[-1]
    if n > target_length:
        return waveform[..., :target_length]
    if n < target_length:
        return torch.nn.functional.pad(waveform, (0, target_length - n))
    return waveform


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance normalization."""
    return (waveform - waveform.mean()) / (waveform.std() + eps)


def preprocess_waveform(
    path: str,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Full waveform preprocessing pipeline for a single audio file.

    Returns (1, target_length) tensor.
    """
    sr = cfg["audio"]["sample_rate"]
    target_length = cfg["audio"]["target_length"]

    waveform, _ = load_audio(path, target_sr=sr)
    waveform = pad_or_trim(waveform, target_length)
    waveform = normalize_waveform(waveform)
    return waveform
