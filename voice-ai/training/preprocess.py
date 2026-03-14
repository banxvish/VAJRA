"""
KAVACHA Voice Deepfake Detection — Audio Preprocessing
=======================================================
Complete audio preprocessing pipeline:
  1. Resample to 16 kHz
  2. Convert to mono
  3. Trim or pad to exactly 2 seconds (32 000 samples)
  4. Normalize waveform (zero-mean, unit-variance)
  5. Generate mel-spectrogram (128 × 128)
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio
import torchaudio.transforms as T


# ------------------------------------------------------------------
# Waveform Utilities
# ------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16_000) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file, resample to *target_sr*, and convert to mono.

    Returns
    -------
    waveform : Tensor of shape (1, num_samples)
    sample_rate : int
    """
    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


def resample_audio(
    waveform: torch.Tensor, orig_sr: int, target_sr: int = 16_000
) -> torch.Tensor:
    """Resample waveform from *orig_sr* to *target_sr*."""
    if orig_sr == target_sr:
        return waveform
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert multi-channel waveform to mono by averaging channels."""
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def pad_or_trim(waveform: torch.Tensor, target_length: int = 32_000) -> torch.Tensor:
    """
    Pad (with zeros) or trim *waveform* to exactly *target_length* samples.
    Expects shape (1, num_samples).
    """
    num_samples = waveform.shape[-1]
    if num_samples > target_length:
        waveform = waveform[..., :target_length]
    elif num_samples < target_length:
        padding = target_length - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    return waveform


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance normalization."""
    mean = waveform.mean()
    std = waveform.std()
    return (waveform - mean) / (std + eps)


# ------------------------------------------------------------------
# Mel-Spectrogram
# ------------------------------------------------------------------

def build_mel_transform(cfg: Dict[str, Any]) -> T.MelSpectrogram:
    """
    Build a ``torchaudio.transforms.MelSpectrogram`` from the YAML config
    section ``spectrogram``.
    """
    spec_cfg = cfg["spectrogram"]
    return T.MelSpectrogram(
        sample_rate=cfg["audio"]["sample_rate"],
        n_fft=spec_cfg["n_fft"],
        hop_length=spec_cfg["hop_length"],
        n_mels=spec_cfg["n_mels"],
        f_min=spec_cfg["f_min"],
        f_max=spec_cfg["f_max"],
    )


def to_mel_spectrogram(
    waveform: torch.Tensor,
    mel_transform: T.MelSpectrogram,
    target_size: Tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """
    Convert waveform → log-mel spectrogram → resized to *target_size*.

    Returns
    -------
    spectrogram : Tensor of shape (1, H, W)   — single channel
    """
    # (1, n_mels, time_frames)
    mel = mel_transform(waveform)

    # Log scale (add eps to avoid log(0))
    mel = torch.log(mel + 1e-9)

    # Resize to exact target_size
    mel = torch.nn.functional.interpolate(
        mel.unsqueeze(0),  # (1, 1, n_mels, time)
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (1, H, W)

    return mel


def spectrogram_to_3ch(spectrogram: torch.Tensor) -> torch.Tensor:
    """
    Replicate single-channel mel spectrogram to 3-channel tensor for
    pretrained ImageNet models.

    Input:  (1, H, W)
    Output: (3, H, W)
    """
    return spectrogram.repeat(3, 1, 1)


# ------------------------------------------------------------------
# Full Pipeline
# ------------------------------------------------------------------

def preprocess_audio(
    path: str,
    cfg: Dict[str, Any],
    mel_transform: Optional[T.MelSpectrogram] = None,
    mode: str = "spectrogram",
) -> torch.Tensor:
    """
    End-to-end preprocessing pipeline for a single audio file.

    Parameters
    ----------
    path : str
        Path to audio file.
    cfg : dict
        Full YAML configuration.
    mel_transform : optional MelSpectrogram transform (built once for speed).
    mode : "spectrogram" or "codec"
        • spectrogram → returns (3, 128, 128) tensor
        • codec        → returns (1, 32000) raw waveform

    Returns
    -------
    Tensor
    """
    target_sr = cfg["audio"]["sample_rate"]
    duration = cfg["audio"]["duration"]
    target_length = target_sr * duration

    waveform, _ = load_audio(path, target_sr=target_sr)
    waveform = pad_or_trim(waveform, target_length=target_length)
    waveform = normalize_waveform(waveform)

    if mode == "codec":
        return waveform  # (1, 32000)

    # Spectrogram path
    if mel_transform is None:
        mel_transform = build_mel_transform(cfg)

    mel = to_mel_spectrogram(waveform, mel_transform)
    mel_3ch = spectrogram_to_3ch(mel)
    return mel_3ch  # (3, 128, 128)
