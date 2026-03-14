"""
KAVACHA Voice Deepfake Detection — Master Training Script
==========================================================
Orchestrates the entire training pipeline:

  1. Load configuration
  2. Prepare datasets & dataloaders
  3. Train spectrogram deepfake detector (EfficientNet-B0, two-stage)
  4. Train codec artifact detector (1D CNN)
  5. Evaluate both models
  6. Export models (PyTorch + ONNX)

Usage
-----
    python train_all.py --config configs/training.yaml --dataset_dir dataset
    python train_all.py --config configs/training.yaml --dataset_dir dataset --skip_codec
    python train_all.py --config configs/training.yaml --dataset_dir dataset --eval_only

All hyperparameters are loaded from the YAML config — nothing is hardcoded.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.utils import load_config, set_seed, get_device, setup_logger, ensure_dirs
from training.augmentations import AugmentationPipeline
from training.dataset import get_spectrogram_dataloaders, get_codec_dataloaders
from training.train_spectrogram import train_spectrogram_model, build_spectrogram_model
from training.train_codec import train_codec_model, CodecDetector
from training.evaluate import evaluate_classifier
from training.export_model import export_all

logger = setup_logger("train_all", log_file="runs/train_all.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KAVACHA Voice Deepfake Detection — Master Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Root dataset directory (overrides config paths.dataset_dir).",
    )
    parser.add_argument(
        "--codec_dataset_dir",
        type=str,
        default=None,
        help="Separate dataset directory for codec detector "
             "(if different from spectrogram dataset).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for exported models (overrides config paths.model_dir).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory for checkpoints (overrides config paths.checkpoint_dir).",
    )
    parser.add_argument(
        "--skip_spectrogram",
        action="store_true",
        help="Skip spectrogram model training.",
    )
    parser.add_argument(
        "--skip_codec",
        action="store_true",
        help="Skip codec model training.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; evaluate and export from existing checkpoints.",
    )
    parser.add_argument(
        "--no_export",
        action="store_true",
        help="Skip ONNX/PyTorch export step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    # ---- Load Configuration ----
    logger.info("Loading configuration...")
    cfg = load_config(args.config)

    # Override paths from CLI
    dataset_dir = args.dataset_dir or cfg["paths"]["dataset_dir"]
    codec_dataset_dir = args.codec_dataset_dir or dataset_dir
    model_dir = args.output_dir or cfg["paths"]["model_dir"]
    checkpoint_dir = args.checkpoint_dir or cfg["paths"]["checkpoint_dir"]
    log_dir = cfg["paths"]["log_dir"]

    ensure_dirs(model_dir, checkpoint_dir, log_dir)

    # ---- Reproducibility ----
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    device = get_device()
    logger.info(f"Device: {device}")

    # ---- Build Augmentation Pipeline ----
    augmentor = AugmentationPipeline(cfg)

    # ==================================================================
    # SPECTROGRAM DEEPFAKE DETECTOR
    # ==================================================================
    spectrogram_model = None

    if not args.skip_spectrogram:
        logger.info("=" * 70)
        logger.info(" SPECTROGRAM DEEPFAKE DETECTOR")
        logger.info("=" * 70)

        # Build dataloaders
        logger.info(f"Loading spectrogram dataset from: {dataset_dir}")
        spec_loaders = get_spectrogram_dataloaders(
            cfg, dataset_dir, augmentation=augmentor
        )

        if "train" not in spec_loaders or "val" not in spec_loaders:
            logger.error(
                "Spectrogram dataset must contain 'train' and 'val' splits."
            )
            sys.exit(1)

        logger.info(
            f"  Train samples: {len(spec_loaders['train'].dataset)}, "
            f"Val samples: {len(spec_loaders['val'].dataset)}"
        )

        if not args.eval_only:
            # Train
            spectrogram_model = train_spectrogram_model(
                cfg,
                train_loader=spec_loaders["train"],
                val_loader=spec_loaders["val"],
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
            )
        else:
            # Load from checkpoint
            import torch as _torch
            best_ckpt = os.path.join(checkpoint_dir, "spectrogram_best.pt")
            if not os.path.exists(best_ckpt):
                logger.error(f"Checkpoint not found: {best_ckpt}")
                sys.exit(1)
            spectrogram_model = build_spectrogram_model(
                num_classes=cfg["training"]["num_classes"],
                dropout=cfg["training"]["dropout"],
                hidden_dim=cfg["training"]["hidden_dim"],
                pretrained=False,
            )
            ckpt = _torch.load(best_ckpt, map_location="cpu", weights_only=False)
            spectrogram_model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded spectrogram model from {best_ckpt}")

        # Evaluate
        if "test" in spec_loaders:
            logger.info("Evaluating spectrogram model on test set...")
            spec_metrics = evaluate_classifier(
                spectrogram_model,
                spec_loaders["test"],
                num_classes=cfg["training"]["num_classes"],
            )
        elif "val" in spec_loaders:
            logger.info("Evaluating spectrogram model on validation set...")
            spec_metrics = evaluate_classifier(
                spectrogram_model,
                spec_loaders["val"],
                num_classes=cfg["training"]["num_classes"],
            )

    # ==================================================================
    # CODEC ARTIFACT DETECTOR
    # ==================================================================
    codec_model = None

    if not args.skip_codec:
        logger.info("=" * 70)
        logger.info(" CODEC ARTIFACT DETECTOR")
        logger.info("=" * 70)

        # Build dataloaders
        logger.info(f"Loading codec dataset from: {codec_dataset_dir}")
        codec_loaders = get_codec_dataloaders(
            cfg, codec_dataset_dir, augmentation=augmentor
        )

        if "train" not in codec_loaders or "val" not in codec_loaders:
            logger.error(
                "Codec dataset must contain 'train' and 'val' splits."
            )
            sys.exit(1)

        logger.info(
            f"  Train samples: {len(codec_loaders['train'].dataset)}, "
            f"Val samples: {len(codec_loaders['val'].dataset)}"
        )

        if not args.eval_only:
            # Train
            codec_model = train_codec_model(
                cfg,
                train_loader=codec_loaders["train"],
                val_loader=codec_loaders["val"],
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
            )
        else:
            # Load from checkpoint
            import torch as _torch
            best_ckpt = os.path.join(checkpoint_dir, "codec_best.pt")
            if not os.path.exists(best_ckpt):
                logger.error(f"Checkpoint not found: {best_ckpt}")
                sys.exit(1)
            codec_model = CodecDetector(
                num_classes=cfg["codec_training"]["num_classes"]
            )
            ckpt = _torch.load(best_ckpt, map_location="cpu", weights_only=False)
            codec_model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded codec model from {best_ckpt}")

        # Evaluate
        if "test" in codec_loaders:
            logger.info("Evaluating codec model on test set...")
            codec_metrics = evaluate_classifier(
                codec_model,
                codec_loaders["test"],
                num_classes=cfg["codec_training"]["num_classes"],
            )
        elif "val" in codec_loaders:
            logger.info("Evaluating codec model on validation set...")
            codec_metrics = evaluate_classifier(
                codec_model,
                codec_loaders["val"],
                num_classes=cfg["codec_training"]["num_classes"],
            )

    # ==================================================================
    # MODEL EXPORT
    # ==================================================================
    if not args.no_export and spectrogram_model is not None and codec_model is not None:
        logger.info("=" * 70)
        logger.info(" MODEL EXPORT (PyTorch + ONNX)")
        logger.info("=" * 70)
        export_all(spectrogram_model, codec_model, cfg, model_dir=model_dir)
    elif not args.no_export:
        # Export whichever model is available
        if spectrogram_model is not None:
            from training.export_model import export_pytorch, export_onnx
            import torch as _torch
            spec_pt = os.path.join(model_dir, "spectrogram_model.pt")
            spec_onnx = os.path.join(model_dir, "spectrogram_model.onnx")
            export_pytorch(spectrogram_model, spec_pt)
            spec_shape = cfg.get("export", {}).get(
                "spectrogram_input_shape", [1, 3, 128, 128]
            )
            export_onnx(
                spectrogram_model,
                _torch.randn(*spec_shape),
                spec_onnx,
                input_names=["mel_spectrogram"],
                output_names=["prediction"],
            )

        if codec_model is not None:
            from training.export_model import export_pytorch, export_onnx
            import torch as _torch
            codec_pt = os.path.join(model_dir, "codec_model.pt")
            codec_onnx = os.path.join(model_dir, "codec_model.onnx")
            export_pytorch(codec_model, codec_pt)
            codec_shape = cfg.get("export", {}).get(
                "codec_input_shape", [1, 32000]
            )
            export_onnx(
                codec_model,
                _torch.randn(codec_shape[0], 1, codec_shape[1]),
                codec_onnx,
                input_names=["raw_waveform"],
                output_names=["prediction"],
            )

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(" TRAINING PIPELINE COMPLETE")
    logger.info(f" Total time: {elapsed / 60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
