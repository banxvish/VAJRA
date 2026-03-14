"""
KAVACHA Voice Deepfake Detection — Data Augmentation Pipeline
=============================================================
Training-time augmentations applied to raw waveforms:
  • Gaussian noise injection
  • Pitch shifting
  • Time stretching
  • Room reverb simulation (convolution with synthetic impulse response)

All augmentations use torchaudio / torch primitives.
Augmentations are applied probabilistically so not every sample is augmented.
"""

from typing import Any, Dict, List, Optional

import torch
import torchaudio.transforms as T
import torchaudio.functional as F


# ------------------------------------------------------------------
# Individual Augmentations
# ------------------------------------------------------------------

class GaussianNoise:
    """Add zero-mean Gaussian noise to the waveform."""

    def __init__(self, std: float = 0.005, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(waveform) * self.std
            waveform = waveform + noise
        return waveform


class PitchShift:
    """
    Shift pitch by a random number of semitones within [-max_semitones, +max_semitones].
    Uses torchaudio.functional.pitch_shift (Sox-based).
    """

    def __init__(self, sample_rate: int = 16_000,
                 max_semitones: int = 2, p: float = 0.5):
        self.sample_rate = sample_rate
        self.max_semitones = max_semitones
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            semitones = (torch.rand(1).item() * 2 - 1) * self.max_semitones
            n_steps = int(round(semitones))
            if n_steps != 0:
                waveform = F.pitch_shift(
                    waveform, self.sample_rate, n_steps
                )
        return waveform


class TimeStretch:
    """
    Stretch/compress the waveform in time using phase vocoder.
    After stretching, the waveform is trimmed/padded to the original length.
    """

    def __init__(self, rates: List[float] = None, n_fft: int = 1024,
                 hop_length: int = 512, p: float = 0.5):
        self.rates = rates or [0.9, 1.1]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            original_length = waveform.shape[-1]
            rate_idx = torch.randint(0, len(self.rates), (1,)).item()
            rate = self.rates[rate_idx]

            # Compute STFT
            spec = torch.stft(
                waveform.squeeze(0),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True,
            )

            # Phase vocoder
            stretcher = T.TimeStretch(
                hop_length=self.hop_length,
                n_freq=spec.shape[-2],
                fixed_rate=rate,
            )
            spec_stretched = stretcher(spec.unsqueeze(0))

            # Inverse STFT
            waveform_stretched = torch.istft(
                spec_stretched.squeeze(0),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

            # Trim or pad back to original length
            if waveform_stretched.shape[-1] > original_length:
                waveform_stretched = waveform_stretched[..., :original_length]
            else:
                diff = original_length - waveform_stretched.shape[-1]
                waveform_stretched = torch.nn.functional.pad(
                    waveform_stretched, (0, diff)
                )

            waveform = waveform_stretched.unsqueeze(0)
        return waveform


class RoomReverb:
    """
    Simulate room reverb by convolving with a simple exponentially
    decaying synthetic impulse response.
    """

    def __init__(self, sample_rate: int = 16_000,
                 delay_ms: float = 20.0, decay: float = 0.4,
                 p: float = 0.3):
        self.sample_rate = sample_rate
        self.delay_ms = delay_ms
        self.decay = decay
        self.p = p

    def _build_impulse_response(self, num_samples: int) -> torch.Tensor:
        """Create a synthetic exponentially decaying impulse response."""
        delay_samples = int(self.sample_rate * self.delay_ms / 1000.0)
        ir_length = min(delay_samples * 10, num_samples)
        ir = torch.zeros(ir_length)
        ir[0] = 1.0

        # Add decaying reflections
        pos = delay_samples
        amplitude = self.decay
        while pos < ir_length and amplitude > 0.01:
            ir[pos] = amplitude
            pos += delay_samples
            amplitude *= self.decay

        return ir

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            num_samples = waveform.shape[-1]
            ir = self._build_impulse_response(num_samples).to(waveform.device)

            # Convolve in frequency domain for efficiency
            original_length = waveform.shape[-1]
            fft_size = 1
            while fft_size < original_length + ir.shape[-1] - 1:
                fft_size *= 2

            wav_fft = torch.fft.rfft(waveform, n=fft_size)
            ir_fft = torch.fft.rfft(ir.unsqueeze(0), n=fft_size)
            convolved = torch.fft.irfft(wav_fft * ir_fft, n=fft_size)
            convolved = convolved[..., :original_length]

            # Normalize to prevent clipping
            peak = convolved.abs().max()
            if peak > 0:
                convolved = convolved / peak * waveform.abs().max()

            waveform = convolved
        return waveform


# ------------------------------------------------------------------
# Composed Augmentation Pipeline
# ------------------------------------------------------------------

class AugmentationPipeline:
    """
    Chain of audio augmentations to apply during training.

    Parameters
    ----------
    cfg : dict
        Full YAML configuration with ``augmentation`` and ``audio`` sections.
    """

    def __init__(self, cfg: Dict[str, Any]):
        aug_cfg = cfg.get("augmentation", {})
        sr = cfg["audio"]["sample_rate"]

        self.transforms: List = [
            GaussianNoise(std=aug_cfg.get("gaussian_noise_std", 0.005)),
            PitchShift(
                sample_rate=sr,
                max_semitones=aug_cfg.get("pitch_shift_semitones", 2),
            ),
            TimeStretch(
                rates=aug_cfg.get("time_stretch_rates", [0.9, 1.1]),
                n_fft=cfg["spectrogram"]["n_fft"],
                hop_length=cfg["spectrogram"]["hop_length"],
            ),
            RoomReverb(
                sample_rate=sr,
                delay_ms=aug_cfg.get("reverb_delay_ms", 20.0),
                decay=aug_cfg.get("reverb_decay", 0.4),
            ),
        ]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            waveform = t(waveform)
        return waveform
