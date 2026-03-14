"""
KAVACHA Voice Deepfake Detection — Dataset & DataLoader
========================================================
PyTorch Dataset classes for:
  1. Spectrogram deepfake detection  (binary: REAL / FAKE)
  2. Codec artifact detection        (3-class: HUMAN / ENCODEC / SOUNDSTREAM)

Folder structure expected::

    dataset/
      train/
        real/   (or human/)
        fake/   (or encodec/, soundstream/)
      val/
        ...
      test/
        ...

Labels are inferred from subfolder names.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio

from training.preprocess import (
    load_audio,
    pad_or_trim,
    normalize_waveform,
    build_mel_transform,
    to_mel_spectrogram,
    spectrogram_to_3ch,
)


# ======================================================================
# Audio file discovery
# ======================================================================

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}


def _find_audio_files(root: str) -> List[Tuple[str, str]]:
    """
    Walk *root* and return list of (filepath, class_name) tuples.
    Class name = immediate parent folder name (lowered).
    """
    samples: List[Tuple[str, str]] = []
    root = Path(root)
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower()
        for fpath in sorted(class_dir.iterdir()):
            if fpath.suffix.lower() in AUDIO_EXTENSIONS:
                samples.append((str(fpath), class_name))
    return samples


# ======================================================================
# Spectrogram Dataset (binary: real / fake)
# ======================================================================

class SpectrogramDeepfakeDataset(Dataset):
    """
    Returns (mel_spectrogram_3ch, label) pairs.

    label mapping:
        real → 0
        fake → 1
    """

    CLASS_MAP = {"real": 0, "fake": 1}

    def __init__(
        self,
        root: str,
        cfg: Dict[str, Any],
        augmentation: Optional[Callable] = None,
    ):
        self.samples = _find_audio_files(root)
        if not self.samples:
            raise RuntimeError(
                f"No audio files found under {root}. "
                "Expected subdirectories: real/ and fake/"
            )
        self.cfg = cfg
        self.augmentation = augmentation
        self.mel_transform = build_mel_transform(cfg)
        self.target_sr = cfg["audio"]["sample_rate"]
        self.target_length = self.target_sr * cfg["audio"]["duration"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, class_name = self.samples[idx]
        label = self.CLASS_MAP[class_name]

        waveform, _ = load_audio(path, target_sr=self.target_sr)
        waveform = pad_or_trim(waveform, target_length=self.target_length)
        waveform = normalize_waveform(waveform)

        # Apply augmentation (raw waveform domain)
        if self.augmentation is not None:
            waveform = self.augmentation(waveform)

        mel = to_mel_spectrogram(waveform, self.mel_transform)
        mel_3ch = spectrogram_to_3ch(mel)  # (3, 128, 128)
        return mel_3ch, label

    def get_labels(self) -> List[int]:
        """Return all labels for sampler construction."""
        return [self.CLASS_MAP[cls] for _, cls in self.samples]


# ======================================================================
# Codec Artifact Dataset (3-class: human / encodec / soundstream)
# ======================================================================

class CodecArtifactDataset(Dataset):
    """
    Returns (raw_waveform, label) pairs.

    label mapping:
        human       → 0
        encodec     → 1
        soundstream → 2
    """

    CLASS_MAP = {"human": 0, "encodec": 1, "soundstream": 2}

    def __init__(
        self,
        root: str,
        cfg: Dict[str, Any],
        augmentation: Optional[Callable] = None,
    ):
        self.samples = _find_audio_files(root)
        if not self.samples:
            raise RuntimeError(
                f"No audio files found under {root}. "
                "Expected subdirectories: human/, encodec/, soundstream/"
            )
        self.cfg = cfg
        self.augmentation = augmentation
        self.target_sr = cfg["audio"]["sample_rate"]
        self.target_length = self.target_sr * cfg["audio"]["duration"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, class_name = self.samples[idx]
        label = self.CLASS_MAP[class_name]

        waveform, _ = load_audio(path, target_sr=self.target_sr)
        waveform = pad_or_trim(waveform, target_length=self.target_length)
        waveform = normalize_waveform(waveform)

        # Apply augmentation (raw waveform domain)
        if self.augmentation is not None:
            waveform = self.augmentation(waveform)

        return waveform, label  # (1, 32000), int

    def get_labels(self) -> List[int]:
        """Return all labels for sampler construction."""
        return [self.CLASS_MAP[cls] for _, cls in self.samples]


# ======================================================================
# DataLoader Factory
# ======================================================================

def _build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to handle class imbalance.
    Each sample is weighted inversely to its class frequency.
    """
    class_counts: Dict[int, int] = {}
    for lbl in labels:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    total = len(labels)
    weights = [total / class_counts[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True,
    )


def get_spectrogram_dataloaders(
    cfg: Dict[str, Any],
    dataset_root: str,
    augmentation: Optional[Callable] = None,
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders for train / val / test splits of the spectrogram
    deepfake detector.
    """
    loaders: Dict[str, DataLoader] = {}
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    for split in ("train", "val", "test"):
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            continue

        aug = augmentation if split == "train" else None
        ds = SpectrogramDeepfakeDataset(split_dir, cfg, augmentation=aug)

        sampler = None
        shuffle = False
        if split == "train":
            sampler = _build_weighted_sampler(ds.get_labels())

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=(split == "train"),
        )

    return loaders


def get_codec_dataloaders(
    cfg: Dict[str, Any],
    dataset_root: str,
    augmentation: Optional[Callable] = None,
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders for train / val / test splits of the codec
    artifact detector.
    """
    loaders: Dict[str, DataLoader] = {}
    batch_size = cfg["codec_training"]["batch_size"]
    num_workers = cfg["codec_training"]["num_workers"]

    for split in ("train", "val", "test"):
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            continue

        aug = augmentation if split == "train" else None
        ds = CodecArtifactDataset(split_dir, cfg, augmentation=aug)

        sampler = None
        shuffle = False
        if split == "train":
            sampler = _build_weighted_sampler(ds.get_labels())

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=(split == "train"),
        )

    return loaders
