"""
KAVACHA Voice Deepfake Detection — Model Export
================================================
Export trained models as:
  • PyTorch state dictionaries (.pt)
  • ONNX graphs (.onnx) with dynamic batch axes
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from training.utils import ensure_dirs, setup_logger

logger = setup_logger("model_export")


# ------------------------------------------------------------------
# PyTorch Export
# ------------------------------------------------------------------

def export_pytorch(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model state_dict as a .pt file.

    Parameters
    ----------
    model : nn.Module
    save_path : str
    metadata : optional dict of extra info to store alongside
    """
    ensure_dirs(os.path.dirname(save_path))

    state = {"model_state_dict": model.state_dict()}
    if metadata:
        state["metadata"] = metadata

    torch.save(state, save_path)
    logger.info(f"PyTorch model saved → {save_path}")


# ------------------------------------------------------------------
# ONNX Export
# ------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    save_path: str,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    opset_version: int = 14,
) -> None:
    """
    Export model to ONNX format with dynamic batch axis.

    Parameters
    ----------
    model : nn.Module
    dummy_input : Tensor matching the model's expected input shape (with batch=1)
    save_path : str
    input_names : list[str]
    output_names : list[str]
    opset_version : int
    """
    ensure_dirs(os.path.dirname(save_path))

    model.eval()

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    dynamic_axes = {
        input_names[0]: {0: "batch_size"},
        output_names[0]: {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    logger.info(f"ONNX model exported → {save_path}")

    # Validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model validation passed ✓")
    except ImportError:
        logger.warning("onnx package not installed — skipping validation.")
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")


# ------------------------------------------------------------------
# Export All Models
# ------------------------------------------------------------------

def export_all(
    spectrogram_model: nn.Module,
    codec_model: nn.Module,
    cfg: Dict[str, Any],
    model_dir: str = "models",
) -> None:
    """
    Export both the spectrogram and codec models in PyTorch and ONNX formats.
    """
    export_cfg = cfg.get("export", {})
    opset = export_cfg.get("opset_version", 14)

    ensure_dirs(model_dir)

    # ---- Spectrogram model (EfficientNet-B0) ----
    spec_pt_path = os.path.join(model_dir, "spectrogram_model.pt")
    spec_onnx_path = os.path.join(model_dir, "spectrogram_model.onnx")

    export_pytorch(
        spectrogram_model,
        spec_pt_path,
        metadata={
            "model_type": "EfficientNet-B0",
            "input_shape": export_cfg.get("spectrogram_input_shape", [1, 3, 128, 128]),
            "num_classes": cfg["training"]["num_classes"],
        },
    )

    spec_input_shape = export_cfg.get("spectrogram_input_shape", [1, 3, 128, 128])
    spec_dummy = torch.randn(*spec_input_shape)
    export_onnx(
        spectrogram_model,
        spec_dummy,
        spec_onnx_path,
        input_names=["mel_spectrogram"],
        output_names=["prediction"],
        opset_version=opset,
    )

    # ---- Codec model (1D CNN) ----
    codec_pt_path = os.path.join(model_dir, "codec_model.pt")
    codec_onnx_path = os.path.join(model_dir, "codec_model.onnx")

    export_pytorch(
        codec_model,
        codec_pt_path,
        metadata={
            "model_type": "CodecDetector-1DCNN",
            "input_shape": export_cfg.get("codec_input_shape", [1, 32000]),
            "num_classes": cfg["codec_training"]["num_classes"],
        },
    )

    codec_input_shape = export_cfg.get("codec_input_shape", [1, 32000])
    # Codec model expects (batch, channels=1, samples)
    codec_dummy = torch.randn(codec_input_shape[0], 1, codec_input_shape[1])
    export_onnx(
        codec_model,
        codec_dummy,
        codec_onnx_path,
        input_names=["raw_waveform"],
        output_names=["prediction"],
        opset_version=opset,
    )

    logger.info("=" * 50)
    logger.info("All models exported successfully ✓")
    logger.info(f"  Spectrogram: {spec_pt_path}, {spec_onnx_path}")
    logger.info(f"  Codec:       {codec_pt_path}, {codec_onnx_path}")
    logger.info("=" * 50)
