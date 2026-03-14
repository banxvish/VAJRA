"""
KAVACHA AI Voice Defense Engine — Hackathon Demo Inference
============================================================
Interactive demo that:
  1. Records microphone audio (or loads a file)
  2. Preprocesses audio
  3. Runs all detection models
  4. Computes ensemble trust score
  5. Displays results in a rich terminal UI

Usage:
    python demo_inference.py                         # record from mic
    python demo_inference.py --file test_audio.wav   # use a file
    python demo_inference.py --file test_audio.wav --enrolled_user alice
"""

import argparse
import os
import sys
import time
import tempfile
import wave
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torchaudio

from utils.checkpoint import load_config, ensure_dirs
from utils.device import get_device
from preprocessing.audio_processor import load_audio, pad_or_trim, normalize_waveform
from preprocessing.spectrogram_generator import build_mel_transform, generate_spectrogram


# ======================================================================
# Terminal Colors
# ======================================================================

class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


# ======================================================================
# Audio Recording
# ======================================================================

def record_audio(duration: float = 2.0, sample_rate: int = 16000) -> torch.Tensor:
    """
    Record audio from the microphone.
    Falls back to generating synthetic audio if pyaudio is unavailable.
    """
    try:
        import pyaudio

        print(f"\n{C.CYAN}🎤 Recording {duration}s of audio...{C.END}")
        print(f"{C.DIM}   Speak now!{C.END}")

        chunk = 1024
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk,
        )

        frames = []
        num_chunks = int(sample_rate * duration / chunk)
        for i in range(num_chunks):
            data = stream.read(chunk)
            frames.append(np.frombuffer(data, dtype=np.float32))

        stream.stop_stream()
        stream.close()
        pa.terminate()

        waveform = np.concatenate(frames)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        print(f"{C.GREEN}   ✓ Recording complete ({waveform.shape[-1]} samples){C.END}")
        return waveform

    except ImportError:
        print(f"\n{C.YELLOW}⚠ pyaudio not installed. Generating synthetic audio for demo.{C.END}")
        n = int(sample_rate * duration)
        t = np.linspace(0, duration, n)
        # Simulate speech-like signal
        signal = (
            0.4 * np.sin(2 * np.pi * 200 * t) +
            0.3 * np.sin(2 * np.pi * 500 * t) +
            0.2 * np.sin(2 * np.pi * 1200 * t) +
            0.05 * np.random.randn(n)
        )
        waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        return waveform


# ======================================================================
# Model Loading
# ======================================================================

def load_spectrogram_model(model_dir: str, cfg: dict):
    """Load trained spectrogram model."""
    from models.spectrogram_model import build_spectrogram_model
    path = os.path.join(model_dir, "spectrogram_model.pt")
    if not os.path.exists(path):
        # Use untrained model for demo
        return build_spectrogram_model(num_classes=2, pretrained=True)
    model = build_spectrogram_model(num_classes=2, pretrained=False)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_wav2vec_model(model_dir: str, cfg: dict):
    """Load trained wav2vec2 model."""
    from models.wav2vec_detector import build_wav2vec_model
    path = os.path.join(model_dir, "wav2vec_detector.pt")
    if not os.path.exists(path):
        return build_wav2vec_model(cfg)
    model = build_wav2vec_model(cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_codec_model(model_dir: str, cfg: dict):
    """Load trained codec model."""
    from models.codec_detector import CodecDetector
    path = os.path.join(model_dir, "codec_model.pt")
    if not os.path.exists(path):
        return CodecDetector(num_classes=3)
    model = CodecDetector(num_classes=3)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ======================================================================
# Inference
# ======================================================================

@torch.no_grad()
def run_spectrogram_inference(model, waveform, cfg):
    """Run spectrogram model on waveform."""
    mel_transform = build_mel_transform(cfg)
    mel = generate_spectrogram(waveform, cfg, mel_transform)
    output = model(mel.unsqueeze(0))
    probs = torch.softmax(output, dim=1).squeeze()
    pred = probs.argmax().item()
    labels = ["REAL", "FAKE"]
    return labels[pred], probs[0].item()  # P(REAL)


@torch.no_grad()
def run_wav2vec_inference(model, waveform, cfg):
    """Run wav2vec2 model on waveform."""
    wav_1d = waveform.squeeze(0)  # (samples,)
    output = model(wav_1d.unsqueeze(0))
    probs = torch.softmax(output, dim=1).squeeze()
    pred = probs.argmax().item()
    labels = ["REAL", "FAKE"]
    return labels[pred], probs[0].item()  # P(REAL)


@torch.no_grad()
def run_codec_inference(model, waveform, cfg):
    """Run codec model on waveform."""
    output = model(waveform.unsqueeze(0))  # (1, 1, 32000)
    probs = torch.softmax(output, dim=1).squeeze()
    pred = probs.argmax().item()
    labels = ["HUMAN", "ENCODEC", "SOUNDSTREAM"]
    return labels[pred], probs.max().item()


def compute_trust_score(spec_real_prob, wav2vec_real_prob, codec_label,
                        speaker_sim, cfg):
    """Compute ensemble trust score."""
    w = cfg["ensemble"]

    # Codec score: 1.0 for HUMAN, low for codec artifacts
    codec_score = 1.0 if codec_label == "HUMAN" else 0.1

    trust = (
        w["spectrogram_weight"] * spec_real_prob +
        w["wav2vec_weight"] * wav2vec_real_prob +
        w["codec_weight"] * codec_score +
        w["speaker_weight"] * max(0.0, speaker_sim)
    )

    thresholds = w["thresholds"]
    if trust >= thresholds["safe"]:
        status = "SAFE"
    elif trust >= thresholds["suspicious"]:
        status = "SUSPICIOUS"
    else:
        status = "FAKE"

    return trust, status


# ======================================================================
# Display
# ======================================================================

def display_results(trust_score, status, spec_label, wav2vec_label,
                    codec_label, speaker_sim):
    """Display results in a rich terminal UI."""

    status_colors = {
        "SAFE": C.GREEN,
        "SUSPICIOUS": C.YELLOW,
        "FAKE": C.RED,
    }
    sc = status_colors.get(status, C.RED)

    print()
    print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════╗{C.END}")
    print(f"{C.BOLD}{C.CYAN}║     🛡️  KAVACHA Voice Defense Engine — Result    ║{C.END}")
    print(f"{C.BOLD}{C.CYAN}╠══════════════════════════════════════════════════╣{C.END}")
    print(f"{C.BOLD}{C.CYAN}║{C.END}")
    print(f"{C.BOLD}{C.CYAN}║{C.END}  {C.BOLD}Trust Score:{C.END}  {sc}{C.BOLD}{trust_score:.2f}{C.END}")
    print(f"{C.BOLD}{C.CYAN}║{C.END}  {C.BOLD}Status:{C.END}       {sc}{C.BOLD}{status}{C.END}")
    print(f"{C.BOLD}{C.CYAN}║{C.END}")
    print(f"{C.BOLD}{C.CYAN}║{C.END}  {C.DIM}── Model Votes ──{C.END}")
    print(f"{C.BOLD}{C.CYAN}║{C.END}  Spectrogram:   {_vote_color(spec_label)}{spec_label}{C.END}")
    print(f"{C.BOLD}{C.CYAN}║{C.END}  Wav2Vec2:      {_vote_color(wav2vec_label)}{wav2vec_label}{C.END}")
    print(f"{C.BOLD}{C.CYAN}║{C.END}  Codec:         {_codec_color(codec_label)}{codec_label}{C.END}")
    print(f"{C.BOLD}{C.CYAN}║{C.END}  Speaker Match: {_sim_color(speaker_sim)}{_sim_label(speaker_sim)}{C.END}")
    print(f"{C.BOLD}{C.CYAN}║{C.END}")
    print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════╝{C.END}")
    print()


def _vote_color(label):
    return C.GREEN if label == "REAL" else C.RED

def _codec_color(label):
    return C.GREEN if label == "HUMAN" else C.RED

def _sim_color(sim):
    if sim >= 0.7: return C.GREEN
    if sim >= 0.4: return C.YELLOW
    return C.RED

def _sim_label(sim):
    if sim >= 0.7: return f"HIGH ({sim:.2f})"
    if sim >= 0.4: return f"MEDIUM ({sim:.2f})"
    return f"LOW ({sim:.2f})"


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KAVACHA Voice Defense — Demo Inference"
    )
    parser.add_argument("--file", type=str, default=None,
                        help="Path to audio file (if not recording)")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--enrolled_user", type=str, default=None,
                        help="User ID for speaker verification")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sr = cfg["audio"]["sample_rate"]
    target_len = cfg["audio"]["target_length"]

    print(f"\n{C.BOLD}{C.HEADER}🛡️  KAVACHA AI Voice Defense Engine{C.END}")
    print(f"{C.DIM}   Advanced Voice Deepfake Detection{C.END}\n")

    # ---- Get audio ----
    if args.file:
        print(f"{C.CYAN}📂 Loading: {args.file}{C.END}")
        waveform, _ = load_audio(args.file, target_sr=sr)
    else:
        waveform = record_audio(duration=cfg["audio"]["duration"],
                                sample_rate=sr)

    waveform = pad_or_trim(waveform, target_len)
    waveform = normalize_waveform(waveform)
    print(f"{C.DIM}   Waveform: {waveform.shape} @ {sr}Hz{C.END}")

    # ---- Load models ----
    print(f"\n{C.CYAN}🔄 Loading models...{C.END}")

    spec_model = load_spectrogram_model(args.model_dir, cfg)
    print(f"   ✓ Spectrogram detector loaded")

    try:
        wav2vec_model = load_wav2vec_model(args.model_dir, cfg)
        print(f"   ✓ Wav2Vec2 detector loaded")
        has_wav2vec = True
    except Exception as e:
        print(f"   ⚠ Wav2Vec2 unavailable: {e}")
        wav2vec_model = None
        has_wav2vec = False

    codec_model = load_codec_model(args.model_dir, cfg)
    print(f"   ✓ Codec detector loaded")

    # ---- Inference ----
    print(f"\n{C.CYAN}🔍 Running inference...{C.END}")

    spec_label, spec_real_prob = run_spectrogram_inference(
        spec_model, waveform, cfg)
    print(f"   Spectrogram: {spec_label} (P(REAL)={spec_real_prob:.3f})")

    if has_wav2vec:
        wav2vec_label, wav2vec_real_prob = run_wav2vec_inference(
            wav2vec_model, waveform, cfg)
        print(f"   Wav2Vec2:    {wav2vec_label} (P(REAL)={wav2vec_real_prob:.3f})")
    else:
        wav2vec_label = spec_label
        wav2vec_real_prob = spec_real_prob

    codec_label, codec_conf = run_codec_inference(
        codec_model, waveform, cfg)
    print(f"   Codec:       {codec_label} (conf={codec_conf:.3f})")

    # ---- Speaker Verification ----
    speaker_sim = 0.0
    if args.enrolled_user:
        try:
            from models.speaker_verification import SpeakerVerifier
            sv = SpeakerVerifier(cfg)
            test_emb = sv.extract_embedding_from_tensor(waveform)
            enrolled_emb = np.load(
                os.path.join(cfg["paths"]["voiceprint_dir"],
                             f"{args.enrolled_user}.npy"))
            speaker_sim = sv.cosine_similarity(enrolled_emb, test_emb)
            print(f"   Speaker:     similarity={speaker_sim:.3f}")
        except Exception as e:
            print(f"   Speaker:     verification failed ({e})")

    # ---- Trust Score ----
    trust_score, status = compute_trust_score(
        spec_real_prob, wav2vec_real_prob, codec_label,
        speaker_sim, cfg,
    )

    # ---- Display ----
    display_results(trust_score, status, spec_label, wav2vec_label,
                    codec_label, speaker_sim)


if __name__ == "__main__":
    main()
