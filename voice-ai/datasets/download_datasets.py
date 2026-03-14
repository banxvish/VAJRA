"""
KAVACHA AI Voice Defense Engine — Dataset Downloader
=====================================================
Automates downloading and extracting benchmark datasets:
  • ASVspoof 2019 LA
  • LibriSpeech dev-clean
  • Mozilla Common Voice (manual)

Downloaded files are cached so repeated runs are fast.
"""

import os
import tarfile
import zipfile
import logging
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _download_file(url: str, dest: str, desc: str = "Downloading") -> str:
    """Download a file with progress reporting."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info(f"  [CACHED] {dest}")
        return str(dest)

    logger.info(f"  {desc}: {url}")
    logger.info(f"  → {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            if block_num % 500 == 0:
                logger.info(f"    {pct:.1f}% ({downloaded // (1024 * 1024)} MB)")

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    logger.info(f"  Download complete.")
    return str(dest)


def _extract_archive(archive_path: str, extract_dir: str) -> None:
    """Extract zip or tar.gz archive."""
    archive = Path(archive_path)
    Path(extract_dir).mkdir(parents=True, exist_ok=True)

    if archive.suffix == ".zip":
        logger.info(f"  Extracting ZIP: {archive}")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extract_dir)
    elif archive.name.endswith(".tar.gz") or archive.name.endswith(".tgz"):
        logger.info(f"  Extracting TAR.GZ: {archive}")
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(extract_dir)
    else:
        logger.warning(f"  Unknown archive format: {archive}")
        return

    logger.info(f"  Extracted to: {extract_dir}")


def download_asvspoof2019(cfg: Dict[str, Any], data_root: str = "downloads") -> Optional[str]:
    """
    Download ASVspoof 2019 LA dataset.

    Note: This is a large dataset (~5 GB). The URL may require
    institutional access. Falls back to creating a placeholder structure
    if the download fails.
    """
    ds_cfg = cfg.get("datasets", {}).get("asvspoof2019", {})
    url = ds_cfg.get("url", "")
    name = ds_cfg.get("name", "ASVspoof2019-LA")

    dest_dir = os.path.join(data_root, name)
    archive_path = os.path.join(data_root, f"{name}.zip")

    if os.path.isdir(dest_dir) and any(Path(dest_dir).iterdir()):
        logger.info(f"  [CACHED] {dest_dir} already exists.")
        return dest_dir

    if url:
        try:
            _download_file(url, archive_path, desc=f"Downloading {name}")
            _extract_archive(archive_path, dest_dir)
            return dest_dir
        except Exception as e:
            logger.warning(f"  Could not download {name}: {e}")
            logger.info("  Creating placeholder dataset structure instead.")

    # Fallback: create minimal placeholder directories
    _create_placeholder_dataset(dest_dir)
    return dest_dir


def download_librispeech(cfg: Dict[str, Any], data_root: str = "downloads") -> Optional[str]:
    """
    Download LibriSpeech dev-clean subset (~350 MB).
    Used as a source of real speech for training data.
    """
    ds_cfg = cfg.get("datasets", {}).get("librispeech", {})
    url = ds_cfg.get("url", "")
    name = ds_cfg.get("name", "LibriSpeech-dev-clean")

    dest_dir = os.path.join(data_root, name)
    archive_path = os.path.join(data_root, f"{name}.tar.gz")

    if os.path.isdir(dest_dir) and any(Path(dest_dir).iterdir()):
        logger.info(f"  [CACHED] {dest_dir} already exists.")
        return dest_dir

    if url:
        try:
            _download_file(url, archive_path, desc=f"Downloading {name}")
            _extract_archive(archive_path, dest_dir)
            return dest_dir
        except Exception as e:
            logger.warning(f"  Could not download {name}: {e}")
            logger.info("  Creating placeholder dataset structure instead.")

    _create_placeholder_dataset(dest_dir)
    return dest_dir


def _create_placeholder_dataset(root: str) -> None:
    """
    Create a minimal placeholder directory structure when real data
    is unavailable. Generates synthetic audio files for pipeline testing.
    """
    import numpy as np

    splits = ["train", "val", "test"]
    classes = ["real", "fake"]

    for split in splits:
        for cls in classes:
            cls_dir = Path(root) / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)

            # Generate 10 synthetic WAV files per class per split
            num_files = 10 if split == "train" else 5
            for i in range(num_files):
                filepath = cls_dir / f"{cls}_{split}_{i:04d}.wav"
                if not filepath.exists():
                    import torchaudio
                    sr = 16000
                    duration = 2.0
                    n_samples = int(sr * duration)

                    if cls == "real":
                        # Simulate speech-like signal with formants
                        t = np.linspace(0, duration, n_samples)
                        signal = (
                            0.5 * np.sin(2 * np.pi * 200 * t) +
                            0.3 * np.sin(2 * np.pi * 500 * t) +
                            0.2 * np.sin(2 * np.pi * 1200 * t) +
                            0.05 * np.random.randn(n_samples)
                        )
                    else:
                        # Simulate artifacts typical of deepfakes
                        t = np.linspace(0, duration, n_samples)
                        signal = (
                            0.4 * np.sin(2 * np.pi * 300 * t) +
                            0.3 * np.sin(2 * np.pi * 800 * t) +
                            0.15 * np.random.randn(n_samples) +
                            0.1 * np.sin(2 * np.pi * 50 * t)  # hum artifact
                        )

                    waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
                    waveform = waveform / waveform.abs().max()
                    torchaudio.save(str(filepath), waveform, sr)

    logger.info(f"  Placeholder dataset created at: {root}")


def download_all_datasets(cfg: Dict[str, Any], data_root: str = "downloads") -> Dict[str, str]:
    """Download all configured datasets. Returns dict of {name: path}."""
    import torch
    logger.info("=" * 60)
    logger.info("  DATASET DOWNLOAD")
    logger.info("=" * 60)

    results = {}

    asvspoof_path = download_asvspoof2019(cfg, data_root)
    if asvspoof_path:
        results["asvspoof2019"] = asvspoof_path

    libri_path = download_librispeech(cfg, data_root)
    if libri_path:
        results["librispeech"] = libri_path

    logger.info(f"  Downloaded {len(results)} datasets.")
    return results
