"""
KAVACHA AI Voice Defense Engine — Spectrogram Generator
========================================================
Mel-spectrogram generation and utilities for the EfficientNet-B0
spectrogram deepfake detector.
"""

from typing import Any, Dict, Tuple

import torch
import torchaudio.transforms as T


def build_mel_transform(cfg: Dict[str, Any]) -> T.MelSpectrogram:
    """Build a MelSpectrogram transform from config."""
    s = cfg["spectrogram"]
    return T.MelSpectrogram(
        sample_rate=cfg["audio"]["sample_rate"],
        n_fft=s["n_fft"],
        hop_length=s["hop_length"],
        n_mels=s["n_mels"],
        f_min=s["f_min"],
        f_max=s["f_max"],
    )


def waveform_to_mel(
    waveform: torch.Tensor,
    mel_transform: T.MelSpectrogram,
    target_size: Tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """
    Convert waveform → log-mel spectrogram → resize to target_size.

    Input:  (1, samples)
    Output: (1, H, W)
    """
    mel = mel_transform(waveform)                        # (1, n_mels, T)
    mel = torch.log(mel + 1e-9)                          # log-scale
    mel = torch.nn.functional.interpolate(
        mel.unsqueeze(0), size=target_size,
        mode="bilinear", align_corners=False,
    ).squeeze(0)                                         # (1, H, W)
    return mel


def mel_to_3ch(mel: torch.Tensor) -> torch.Tensor:
    """Replicate single-channel mel → 3-channel for ImageNet models."""
    return mel.repeat(3, 1, 1)                           # (3, H, W)


def generate_spectrogram(
    waveform: torch.Tensor,
    cfg: Dict[str, Any],
    mel_transform: T.MelSpectrogram = None,
) -> torch.Tensor:
    """
    Full pipeline: waveform → 3-channel log-mel spectrogram (3, 128, 128).
    """
    if mel_transform is None:
        mel_transform = build_mel_transform(cfg)

    size = cfg["spectrogram"]["image_size"]
    mel = waveform_to_mel(waveform, mel_transform, target_size=(size, size))
    return mel_to_3ch(mel)
