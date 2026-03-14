"""
KAVACHA AI Voice Defense Engine — Dataset Builder
===================================================
PyTorch Dataset classes and DataLoader factories for:
  1. Spectrogram deepfake detection   (binary: real / fake)
  2. Wav2Vec2 deepfake detection      (binary: real / fake, raw waveform)
  3. Codec artifact detection          (3-class: human / encodec / soundstream)

Labels inferred from folder names. Weighted sampling for class imbalance.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from preprocessing.audio_processor import load_audio, pad_or_trim, normalize_waveform
from preprocessing.spectrogram_generator import build_mel_transform, generate_spectrogram

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}


def _find_audio_files(root: str) -> List[Tuple[str, str]]:
    """Walk root → list of (filepath, class_name) tuples."""
    samples = []
    root = Path(root)
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        cls = class_dir.name.lower()
        for fp in sorted(class_dir.iterdir()):
            if fp.suffix.lower() in AUDIO_EXTENSIONS:
                samples.append((str(fp), cls))
    return samples


# ==================================================================
# Spectrogram Dataset
# ==================================================================

class SpectrogramDataset(Dataset):
    """Returns (mel_3ch, label) — for EfficientNet-B0 detector."""

    CLASS_MAP = {"real": 0, "fake": 1}

    def __init__(self, root: str, cfg: Dict[str, Any],
                 augmentation: Optional[Callable] = None):
        self.samples = _find_audio_files(root)
        if not self.samples:
            raise RuntimeError(f"No audio in {root}. Need real/ + fake/ sub-dirs.")
        self.cfg = cfg
        self.aug = augmentation
        self.mel_transform = build_mel_transform(cfg)
        self.sr = cfg["audio"]["sample_rate"]
        self.target_len = cfg["audio"]["target_length"]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        label = self.CLASS_MAP[cls]
        wav, _ = load_audio(path, self.sr)
        wav = pad_or_trim(wav, self.target_len)
        wav = normalize_waveform(wav)
        if self.aug and cls == "real":
            wav = self.aug(wav)
        mel = generate_spectrogram(wav, self.cfg, self.mel_transform)
        return mel, label

    def get_labels(self):
        return [self.CLASS_MAP[c] for _, c in self.samples]


# ==================================================================
# Wav2Vec2 Waveform Dataset
# ==================================================================

class WaveformDataset(Dataset):
    """Returns (waveform_1d, label) — for Wav2Vec2 detector."""

    CLASS_MAP = {"real": 0, "fake": 1}

    def __init__(self, root: str, cfg: Dict[str, Any],
                 augmentation: Optional[Callable] = None):
        self.samples = _find_audio_files(root)
        if not self.samples:
            raise RuntimeError(f"No audio in {root}.")
        self.cfg = cfg
        self.aug = augmentation
        self.sr = cfg["audio"]["sample_rate"]
        self.target_len = cfg["audio"]["target_length"]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        label = self.CLASS_MAP[cls]
        wav, _ = load_audio(path, self.sr)
        wav = pad_or_trim(wav, self.target_len)
        wav = normalize_waveform(wav)
        if self.aug and cls == "real":
            wav = self.aug(wav)
        # Wav2Vec2 expects (samples,) — squeeze channel dim
        return wav.squeeze(0), label

    def get_labels(self):
        return [self.CLASS_MAP[c] for _, c in self.samples]


# ==================================================================
# Codec Artifact Dataset
# ==================================================================

class CodecDataset(Dataset):
    """Returns (waveform, label) — for 1D CNN codec detector."""

    CLASS_MAP = {"human": 0, "real": 0, "encodec": 1, "soundstream": 2, "fake": 1}

    def __init__(self, root: str, cfg: Dict[str, Any],
                 augmentation: Optional[Callable] = None):
        self.samples = _find_audio_files(root)
        if not self.samples:
            raise RuntimeError(f"No audio in {root}.")
        self.cfg = cfg
        self.aug = augmentation
        self.sr = cfg["audio"]["sample_rate"]
        self.target_len = cfg["audio"]["target_length"]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        label = self.CLASS_MAP.get(cls, 0)
        wav, _ = load_audio(path, self.sr)
        wav = pad_or_trim(wav, self.target_len)
        wav = normalize_waveform(wav)
        if self.aug and cls in ("human", "real"):
            wav = self.aug(wav)
        return wav, label

    def get_labels(self):
        return [self.CLASS_MAP.get(c, 0) for _, c in self.samples]


# ==================================================================
# DataLoader Factory
# ==================================================================

def _weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    total = len(labels)
    weights = [total / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=total, replacement=True)


def build_dataloaders(
    dataset_cls,
    cfg: Dict[str, Any],
    dataset_root: str,
    augmentation: Optional[Callable] = None,
    batch_size: int = 32,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders for any dataset class."""
    dl_cfg = cfg.get("dataloader", {})
    num_workers = dl_cfg.get("num_workers", 4)
    pin = dl_cfg.get("pin_memory", True)
    persistent = dl_cfg.get("persistent_workers", True) and num_workers > 0

    loaders = {}
    for split in ("train", "val", "test"):
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            continue

        aug = augmentation if split == "train" else None
        ds = dataset_cls(split_dir, cfg, augmentation=aug)

        sampler = _weighted_sampler(ds.get_labels()) if split == "train" else None

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            drop_last=(split == "train"),
        )

    return loaders
