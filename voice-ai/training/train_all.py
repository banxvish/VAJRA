"""
KAVACHA AI Voice Defense Engine — Master Training Pipeline
============================================================
Single entry point that orchestrates:

  1. Dataset download / preparation
  2. Audio preprocessing + augmentation setup
  3. Spectrogram model training (EfficientNet-B0, two-stage)
  4. Wav2Vec2 detector training (two-stage transfer learning)
  5. Codec detector training (1D CNN)
  6. Evaluation of all models
  7. Model export (PyTorch + ONNX)
  8. Experiment metadata logging

Usage:
    python training/train_all.py
    python training/train_all.py --config configs/training.yaml
    python training/train_all.py --skip_wav2vec
    python training/train_all.py --eval_only
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.seed import set_seed
from utils.device import get_device, device_info
from utils.checkpoint import load_config, ensure_dirs, setup_logger

from datasets.download_datasets import download_all_datasets
from datasets.dataset_builder import (
    SpectrogramDataset, WaveformDataset, CodecDataset, build_dataloaders,
)
from preprocessing.augmentations import AugmentationPipeline

from training.train_spectrogram import train_spectrogram
from training.train_wav2vec import train_wav2vec
from training.train_codec import train_codec

from evaluation.evaluate_models import evaluate_all_models
from export.export_models import export_all_models
from experiments.experiment_logger import ExperimentLogger

logger = setup_logger("train_all", log_file="runs/train_all.log")


def parse_args():
    p = argparse.ArgumentParser(description="KAVACHA Training Pipeline")
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--codec_dataset_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--skip_spectrogram", action="store_true")
    p.add_argument("--skip_wav2vec", action="store_true")
    p.add_argument("--skip_codec", action="store_true")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--no_export", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # ---- Configuration ----
    cfg = load_config(args.config)
    dataset_dir = args.dataset_dir or cfg["paths"]["dataset_dir"]
    codec_dir = args.codec_dataset_dir or dataset_dir
    model_dir = args.output_dir or cfg["paths"]["model_dir"]
    ckpt_dir = args.checkpoint_dir or cfg["paths"]["checkpoint_dir"]
    log_dir = cfg["paths"]["log_dir"]

    ensure_dirs(model_dir, ckpt_dir, log_dir)
    set_seed(cfg["seed"])

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   KAVACHA AI Voice Defense Engine — Training Pipeline    ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"Device: {device_info()}")
    logger.info(f"Seed: {cfg['seed']}")

    # ---- Experiment Logger ----
    exp = ExperimentLogger(log_dir=log_dir, experiment_name="kavacha_training")
    exp.log_config(cfg)

    # ---- Dataset Download ----
    if not args.skip_download:
        logger.info("\n▸ STEP 1: DATASET DOWNLOAD")
        download_all_datasets(cfg, data_root="downloads")

        # Use placeholder dataset if real data dir doesn't have splits
        if not os.path.isdir(os.path.join(dataset_dir, "train")):
            logger.info("  No train/ split found in dataset_dir, using downloaded data.")
            alt = os.path.join("downloads", "ASVspoof2019-LA")
            if os.path.isdir(os.path.join(alt, "train")):
                dataset_dir = alt
                codec_dir = alt

    # ---- Augmentation ----
    augmentor = AugmentationPipeline(cfg)

    # ──────────────────────────────────────────────────────────────
    # MODEL 1: SPECTROGRAM DEEPFAKE DETECTOR
    # ──────────────────────────────────────────────────────────────
    spec_model = None
    spec_loaders = {}

    if not args.skip_spectrogram:
        logger.info("\n▸ STEP 2: SPECTROGRAM DEEPFAKE DETECTOR")
        spec_loaders = build_dataloaders(
            SpectrogramDataset, cfg, dataset_dir,
            augmentation=augmentor,
            batch_size=cfg["training"]["batch_size"],
        )
        logger.info(f"  Train: {len(spec_loaders.get('train', {}).dataset)} samples")

        if not args.eval_only:
            spec_model = train_spectrogram(
                cfg, spec_loaders["train"], spec_loaders["val"],
                checkpoint_dir=ckpt_dir, log_dir=log_dir,
            )
            exp.log_model_info("spectrogram", {
                "architecture": "EfficientNet-B0",
                "input_shape": cfg["export"]["spectrogram_input_shape"],
            })
        else:
            from models.spectrogram_model import build_spectrogram_model
            spec_model = build_spectrogram_model(
                num_classes=cfg["training"]["num_classes"], pretrained=False)
            ckpt = os.path.join(ckpt_dir, "spectrogram_best.pt")
            import torch
            spec_model.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"])

    # ──────────────────────────────────────────────────────────────
    # MODEL 2: WAV2VEC2 SELF-SUPERVISED DETECTOR
    # ──────────────────────────────────────────────────────────────
    wav2vec_model = None
    wav2vec_loaders = {}

    if not args.skip_wav2vec:
        logger.info("\n▸ STEP 3: WAV2VEC2 DEEPFAKE DETECTOR")
        wav2vec_loaders = build_dataloaders(
            WaveformDataset, cfg, dataset_dir,
            augmentation=augmentor,
            batch_size=cfg["wav2vec"]["batch_size"],
        )
        logger.info(f"  Train: {len(wav2vec_loaders.get('train', {}).dataset)} samples")

        if not args.eval_only:
            wav2vec_model = train_wav2vec(
                cfg, wav2vec_loaders["train"], wav2vec_loaders["val"],
                checkpoint_dir=ckpt_dir, log_dir=log_dir,
            )
            exp.log_model_info("wav2vec", {
                "architecture": cfg["wav2vec"]["model_name"],
                "input_shape": cfg["export"]["wav2vec_input_shape"],
            })
        else:
            from models.wav2vec_detector import build_wav2vec_model
            wav2vec_model = build_wav2vec_model(cfg)
            ckpt = os.path.join(ckpt_dir, "wav2vec_best.pt")
            import torch
            wav2vec_model.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"])

    # ──────────────────────────────────────────────────────────────
    # MODEL 3: CODEC ARTIFACT DETECTOR
    # ──────────────────────────────────────────────────────────────
    codec_model = None
    codec_loaders = {}

    if not args.skip_codec:
        logger.info("\n▸ STEP 4: CODEC ARTIFACT DETECTOR")
        codec_loaders = build_dataloaders(
            CodecDataset, cfg, codec_dir,
            augmentation=augmentor,
            batch_size=cfg["codec_training"]["batch_size"],
        )
        logger.info(f"  Train: {len(codec_loaders.get('train', {}).dataset)} samples")

        if not args.eval_only:
            codec_model = train_codec(
                cfg, codec_loaders["train"], codec_loaders["val"],
                checkpoint_dir=ckpt_dir, log_dir=log_dir,
            )
            exp.log_model_info("codec", {
                "architecture": "1D-CNN CodecDetector",
                "input_shape": cfg["export"]["codec_input_shape"],
            })
        else:
            from models.codec_detector import CodecDetector
            codec_model = CodecDetector(num_classes=cfg["codec_training"]["num_classes"])
            ckpt = os.path.join(ckpt_dir, "codec_best.pt")
            import torch
            codec_model.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"])

    # ──────────────────────────────────────────────────────────────
    # EVALUATION
    # ──────────────────────────────────────────────────────────────
    logger.info("\n▸ STEP 5: EVALUATION")
    results = evaluate_all_models(
        spectrogram_model=spec_model,
        wav2vec_model=wav2vec_model,
        codec_model=codec_model,
        spec_loader=spec_loaders.get("test") or spec_loaders.get("val"),
        wav2vec_loader=wav2vec_loaders.get("test") or wav2vec_loaders.get("val"),
        codec_loader=codec_loaders.get("test") or codec_loaders.get("val"),
        cfg=cfg,
    )
    for model_name, metrics in results.items():
        exp.log_metrics(model_name, metrics)

    # ──────────────────────────────────────────────────────────────
    # EXPORT
    # ──────────────────────────────────────────────────────────────
    if not args.no_export:
        logger.info("\n▸ STEP 6: MODEL EXPORT")
        export_all_models(spec_model, wav2vec_model, codec_model, cfg, model_dir)

    # ──────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    exp.finalize("completed")

    logger.info("\n╔" + "═" * 58 + "╗")
    logger.info("║   ✓ KAVACHA TRAINING PIPELINE COMPLETE                  ║")
    logger.info(f"║   Total time: {elapsed / 60:>6.1f} minutes{' ' * 29}║")
    logger.info("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
