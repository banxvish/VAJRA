"""
KAVACHA AI Voice Defense Engine — Wav2Vec2 ONNX Export
======================================================
Exports the fine-tuned HuggingFace Wav2Vec2 detector into an 
optimized ONNX graph for high-speed CPU/Edge inference.
"""

import os
import torch
from transformers import Wav2Vec2ForSequenceClassification

def export_wav2vec_onnx(model_dir: str, output_path: str):
    print(f"Loading Fine-Tuned Model from {model_dir}...")
    
    # Load PyTorch Fine-Tuned Model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    # Create Dummy Input matching the Processor output (batch=1, samples=32000)
    # 2 seconds at 16kHz
    dummy_input = torch.randn(1, 32000, dtype=torch.float32)

    print(f"Exporting model to ONNX format -> {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"}, 
            "logits": {0: "batch_size"}
        }
    )
    
    print(f"✅ Export Successful! File saved to: {output_path}")

if __name__ == "__main__":
    export_wav2vec_onnx(model_dir="export/wav2vec_finetuned_hf", output_path="export/onnx/wav2vec2_classifier.onnx")
