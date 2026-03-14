"""
KAVACHA AI Voice Defense Engine — Model Export
================================================
Export trained models as:
  • PyTorch state_dict (.pt)
  • ONNX (.onnx) with dynamic batch axis and validation
"""

import os
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from utils.checkpoint import ensure_dirs

logger = logging.getLogger(__name__)


def export_pytorch(model: nn.Module, path: str,
                   metadata: Optional[Dict] = None) -> None:
    """Save model state_dict."""
    ensure_dirs(os.path.dirname(path))
    state = {"model_state_dict": model.state_dict()}
    if metadata:
        state["metadata"] = metadata
    torch.save(state, path)
    logger.info(f"PyTorch model → {path}")


def export_onnx(model: nn.Module, dummy_input: torch.Tensor, path: str,
                input_names=None, output_names=None, opset: int = 14) -> None:
    """Export to ONNX with dynamic batch axis."""
    ensure_dirs(os.path.dirname(path))
    model.eval()

    inp = input_names or ["input"]
    out = output_names or ["output"]
    dynamic = {inp[0]: {0: "batch"}, out[0]: {0: "batch"}}

    torch.onnx.export(
        model, dummy_input, path,
        export_params=True, opset_version=opset,
        do_constant_folding=True,
        input_names=inp, output_names=out,
        dynamic_axes=dynamic,
    )
    logger.info(f"ONNX model → {path}")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        logger.info("ONNX validation ✓")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"ONNX validation issue: {e}")


def export_all_models(
    spectrogram_model: Optional[nn.Module],
    wav2vec_model: Optional[nn.Module],
    codec_model: Optional[nn.Module],
    cfg: Dict[str, Any],
    model_dir: str = "models",
) -> None:
    """Export all available models in PyTorch + ONNX formats."""
    ensure_dirs(model_dir)
    opset = cfg.get("export", {}).get("opset_version", 14)

    if spectrogram_model:
        export_pytorch(spectrogram_model,
                       os.path.join(model_dir, "spectrogram_model.pt"),
                       metadata={"type": "EfficientNet-B0"})
        shape = cfg.get("export", {}).get("spectrogram_input_shape", [1, 3, 128, 128])
        export_onnx(spectrogram_model, torch.randn(*shape),
                    os.path.join(model_dir, "spectrogram_model.onnx"),
                    input_names=["mel_spectrogram"], output_names=["prediction"],
                    opset=opset)

    if wav2vec_model:
        export_pytorch(wav2vec_model,
                       os.path.join(model_dir, "wav2vec_detector.pt"),
                       metadata={"type": "Wav2Vec2-Base"})
        shape = cfg.get("export", {}).get("wav2vec_input_shape", [1, 32000])
        export_onnx(wav2vec_model, torch.randn(*shape),
                    os.path.join(model_dir, "wav2vec_detector.onnx"),
                    input_names=["raw_waveform"], output_names=["prediction"],
                    opset=opset)

    if codec_model:
        export_pytorch(codec_model,
                       os.path.join(model_dir, "codec_model.pt"),
                       metadata={"type": "CodecDetector-1DCNN"})
        shape = cfg.get("export", {}).get("codec_input_shape", [1, 32000])
        export_onnx(codec_model, torch.randn(shape[0], 1, shape[1]),
                    os.path.join(model_dir, "codec_model.onnx"),
                    input_names=["raw_waveform"], output_names=["prediction"],
                    opset=opset)

    logger.info("=" * 50)
    logger.info("All models exported ✓")
    logger.info("=" * 50)
