"""
KAVACHA AI Voice Defense Engine — Data Augmentations
=====================================================
Training-time augmentations applied to raw waveforms:
  • Gaussian noise injection
  • Pitch shifting
  • Time stretching (phase vocoder)
  • Background noise mixing
  • Room reverberation (synthetic IR convolution)

All augmentations are probabilistic — not every sample is augmented.
Applied to real speech samples only (configurable).
"""

from typing import Any, Dict, List

import torch
import torchaudio.functional as F
import torchaudio.transforms as T


class GaussianNoise:
    """Add zero-mean Gaussian noise."""

    def __init__(self, std: float = 0.005, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            waveform = waveform + torch.randn_like(waveform) * self.std
        return waveform


class PitchShift:
    """Shift pitch by random semitones."""

    def __init__(self, sample_rate: int = 16_000,
                 max_semitones: int = 2, p: float = 0.5):
        self.sr = sample_rate
        self.max = max_semitones
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            n = int(round((torch.rand(1).item() * 2 - 1) * self.max))
            if n != 0:
                waveform = F.pitch_shift(waveform, self.sr, n)
        return waveform


class TimeStretch:
    """Stretch/compress time using phase vocoder, then trim/pad to original length."""

    def __init__(self, rates: List[float] = None,
                 n_fft: int = 1024, hop_length: int = 512, p: float = 0.5):
        self.rates = rates or [0.9, 1.1]
        self.n_fft = n_fft
        self.hop = hop_length
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            orig_len = waveform.shape[-1]
            rate = self.rates[torch.randint(len(self.rates), (1,)).item()]

            spec = torch.stft(waveform.squeeze(0), n_fft=self.n_fft,
                              hop_length=self.hop, return_complex=True)
            stretcher = T.TimeStretch(hop_length=self.hop,
                                      n_freq=spec.shape[-2], fixed_rate=rate)
            spec_s = stretcher(spec.unsqueeze(0))
            wav = torch.istft(spec_s.squeeze(0), n_fft=self.n_fft,
                              hop_length=self.hop)

            if wav.shape[-1] > orig_len:
                wav = wav[..., :orig_len]
            else:
                wav = torch.nn.functional.pad(wav, (0, orig_len - wav.shape[-1]))
            waveform = wav.unsqueeze(0)
        return waveform


class BackgroundNoise:
    """Mix with random noise at a specified SNR (dB)."""

    def __init__(self, snr_db: float = 15.0, p: float = 0.3):
        self.snr_db = snr_db
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(waveform)
            signal_power = waveform.pow(2).mean()
            noise_power = noise.pow(2).mean()
            snr_linear = 10 ** (self.snr_db / 10)
            scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-8))
            waveform = waveform + scale * noise
        return waveform


class RoomReverb:
    """Convolve with a synthetic exponentially decaying impulse response."""

    def __init__(self, sample_rate: int = 16_000,
                 delay_ms: float = 20.0, decay: float = 0.4, p: float = 0.3):
        self.sr = sample_rate
        self.delay_ms = delay_ms
        self.decay = decay
        self.p = p

    def _build_ir(self, n: int) -> torch.Tensor:
        delay = int(self.sr * self.delay_ms / 1000)
        ir_len = min(delay * 10, n)
        ir = torch.zeros(ir_len)
        ir[0] = 1.0
        pos, amp = delay, self.decay
        while pos < ir_len and amp > 0.01:
            ir[pos] = amp
            pos += delay
            amp *= self.decay
        return ir

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            orig_len = waveform.shape[-1]
            ir = self._build_ir(orig_len).to(waveform.device)
            fft_size = 1
            while fft_size < orig_len + ir.shape[-1] - 1:
                fft_size *= 2
            conv = torch.fft.irfft(
                torch.fft.rfft(waveform, n=fft_size) *
                torch.fft.rfft(ir.unsqueeze(0), n=fft_size),
                n=fft_size,
            )[..., :orig_len]
            peak = conv.abs().max()
            if peak > 0:
                conv = conv / peak * waveform.abs().max()
            waveform = conv
        return waveform


class AugmentationPipeline:
    """Composable chain of audio augmentations."""

    def __init__(self, cfg: Dict[str, Any]):
        aug = cfg.get("augmentation", {})
        sr = cfg["audio"]["sample_rate"]
        spec = cfg["spectrogram"]

        self.transforms = [
            GaussianNoise(std=aug.get("gaussian_noise_std", 0.005)),
            PitchShift(sample_rate=sr,
                       max_semitones=aug.get("pitch_shift_semitones", 2)),
            TimeStretch(rates=aug.get("time_stretch_rates", [0.9, 1.1]),
                        n_fft=spec["n_fft"], hop_length=spec["hop_length"]),
            BackgroundNoise(snr_db=aug.get("bg_noise_snr_db", 15)),
            RoomReverb(sample_rate=sr,
                       delay_ms=aug.get("reverb_delay_ms", 20),
                       decay=aug.get("reverb_decay", 0.4)),
        ]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            waveform = t(waveform)
        return waveform
