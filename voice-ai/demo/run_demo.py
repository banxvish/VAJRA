"""
KAVACHA AI Voice Defense Engine — Deployment Demo Pipeline
===========================================================
Loads KAVACHA system securely and evaluates input via CLI in < 2 seconds.

Usage:
  python demo/run_demo.py --file path/to/suspicious.wav
"""

import argparse
import sys
import time
from pathlib import Path

# Fix sys.path to run cleanly from root
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Optional
import numpy as np

from utils.checkpoint import load_config
from inference.model_loader import ModelLoader
from inference.audio_pipeline import AudioPipeline
from inference.ensemble_engine import EnsembleEngine
from utils.visualization import console_trust_visualizer
from utils.device import get_device
from utils.seed import set_seed

def main():
    parser = argparse.ArgumentParser(description="KAVACHA Deployment CLI")
    parser.add_argument("--file", type=str, required=True, help="Audio file to analyze")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--enrolled_user", type=str, default=None, help="Check 1-v-1 similarity")
    args = parser.parse_args()

    t0 = time.time()
    
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    
    # Initialization
    models = ModelLoader(cfg, model_dir="models", device=get_device())
    models.load_all()
    
    audio_pipe = AudioPipeline(cfg)
    engine = EnsembleEngine(models, cfg)
    
    # Process
    try:
        print(f"Analyzing {Path(args.file).name}...")
        waveform, spectrogram = audio_pipe.process_file(args.file)
        
        user_emb = None
        if args.enrolled_user:
            vp_path = Path(cfg["paths"]["voiceprint_dir"]) / f"{args.enrolled_user}.npy"
            if vp_path.exists():
                user_emb = np.load(str(vp_path))
            else:
                print(f"Warning: Enrolled profile {vp_path} not found.")

        result = engine.evaluate(waveform, spectrogram, enrolled_user_emb=user_emb)
        
        console_trust_visualizer(result)
        
    except Exception as e:
        print(f"Error processing audio: {e}")

    print(f"Latency: {time.time()-t0:.2f} seconds")

if __name__ == "__main__":
    main()
