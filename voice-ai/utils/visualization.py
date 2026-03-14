"""
KAVACHA AI Voice Defense Engine — Visualization Utilities
===========================================================
Methods to beautifully render Trust Scores in console or Streamlit.
"""

import logging
from typing import Dict, Any

def console_trust_visualizer(results: Dict[str, Any]):
    """Pretty prints the ensemble dictionary beautifully in terminal."""
    
    C_RED = "\033[91m"
    C_GREEN = "\033[92m"
    C_YELLOW = "\033[93m"
    C_BOLD = "\033[1m"
    C_END = "\033[0m"

    st_color = {
        "SAFE": C_GREEN,
        "SUSPICIOUS": C_YELLOW,
        "FAKE": C_RED
    }
    col = st_color.get(results['status'], C_RED)

    def colorize_label(label, good_val):
        return C_GREEN if label == good_val else C_RED

    print()
    print(f"KAVACHA Analysis Result")
    print(f"=======================\n")
    print(f"Trust Score: {col}{C_BOLD}{results['trust_score']:.2f}{C_END}")
    print(f"Status:      {col}{C_BOLD}{results['status']}{C_END}\n")
    print("Model Votes")
    print(f"Spectrogram:   {colorize_label(results['spectrogram_vote'], 'REAL')}{results['spectrogram_vote']}{C_END}")
    print(f"Wav2Vec2:      {colorize_label(results['wav2vec_vote'], 'REAL')}{results['wav2vec_vote']}{C_END}")
    print(f"Codec:         {colorize_label(results['codec_vote'], 'HUMAN')}{results['codec_vote']}{C_END}")

    sim = results['speaker_sim']
    sim_txt = "HIGH" if sim >= 0.7 else ("MEDIUM" if sim >= 0.4 else "LOW")
    sim_col = C_GREEN if sim_txt == "HIGH" else (C_YELLOW if sim_txt == "MEDIUM" else C_RED)
    print(f"Speaker Match: {sim_col}{sim_txt}{C_END} ({sim:.2f})")
    print()
