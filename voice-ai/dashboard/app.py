"""
KAVACHA AI Voice Defense Engine — Streamlit Web Dashboard
=============================================================
Interactive UI for real-time model evaluation and visualizations.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
import torch

from utils.checkpoint import load_config
from utils.device import get_device
from utils.seed import set_seed
from utils.audio_utils import save_uploaded_audio, cleanup_temp_file
from inference.model_loader import ModelLoader
from inference.audio_pipeline import AudioPipeline
from inference.ensemble_engine import EnsembleEngine

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="KAVACHA AI Defense",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    .trust-safe { color: #2ecc71; font-weight: bold; font-size: 2em; text-align: center; }
    .trust-sus  { color: #f1c40f; font-weight: bold; font-size: 2em; text-align: center; }
    .trust-fake { color: #e74c3c; font-weight: bold; font-size: 2em; text-align: center; }
    .metric-val { font-size: 1.5em; font-weight: bold; }
    .metric-ok  { color: #2ecc71; }
    .metric-bad { color: #e74c3c; }
    </style>
""", unsafe_allow_html=True)


# ---- CACHING MODELS ----
@st.cache_resource(show_spinner=False)
def load_kavacha_system():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs/training.yaml"))
    set_seed(cfg["seed"])
    
    device = get_device()
    models = ModelLoader(cfg, model_dir=os.path.join(PROJECT_ROOT, "models"), device=device)
    models.load_all()
    
    pipe = AudioPipeline(cfg)
    engine = EnsembleEngine(models, cfg)
    return cfg, pipe, engine


# ---- UI UTILS ----
def display_results(results: dict):
    st.markdown("### Trust Score Visualization")
    
    status = results["status"]
    score = results["trust_score"]
    
    if status == "SAFE":
        st.markdown(f"<div class='trust-safe'>🛡️ SAFE ({score:.2f})</div>", unsafe_allow_html=True)
        st.progress(score, text="Verified Human Identity")
    elif status == "SUSPICIOUS":
        st.markdown(f"<div class='trust-sus'>⚠️ SUSPICIOUS ({score:.2f})</div>", unsafe_allow_html=True)
        st.progress(score, text="Potential Generative Traces")
    else:
        st.markdown(f"<div class='trust-fake'>🛑 FAKE ({score:.2f})</div>", unsafe_allow_html=True)
        st.progress(score, text="Deepfake Artifacts Detected")

    st.markdown("---")
    st.markdown("### Model Results")

    col1, col2, col3, col4 = st.columns(4)

    def _color(val, ok_val):
        return "metric-ok" if val == ok_val else "metric-bad"

    with col1:
        v = results["spectrogram_vote"]
        c = _color(v, "REAL")
        st.markdown(f"**Spectrogram**<br><span class='metric-val {c}'>{v}</span>", unsafe_allow_html=True)

    with col2:
        v = results["wav2vec_vote"]
        c = _color(v, "REAL")
        st.markdown(f"**Wav2Vec2**<br><span class='metric-val {c}'>{v}</span>", unsafe_allow_html=True)

    with col3:
        v = results["codec_vote"]
        c = "metric-ok" if v == "HUMAN" else "metric-bad"
        st.markdown(f"**Codec**<br><span class='metric-val {c}'>{v}</span>", unsafe_allow_html=True)

    with col4:
        s = results["speaker_sim"]
        if s >= 0.7: txt, c = "HIGH", "metric-ok"
        elif s >= 0.4: txt, c = "MEDIUM", "trust-sus"
        else: txt, c = "LOW", "metric-bad"
        st.markdown(f"**Speaker**<br><span class='metric-val {c}'>{txt} ({s:.2f})</span>", unsafe_allow_html=True)


# ---- APP LOOP ----
def main():
    st.title("🛡️ KAVACHA AI Voice Defense Engine")
    st.write("Detect voice deepfakes, neural codecs, and unauthorized speaker impersonations in real-time.")

    cfg, audio_pipe, ensemble_engine = load_kavacha_system()

    st.markdown("---")
    st.subheader("1. Audio Input")
    
    # 2 sections for audio input
    tab1, tab2 = st.tabs(["Upload Audio File", "Record Microphone"])
    
    audio_file = None
    
    with tab1:
        uploaded_file = st.file_uploader("Select an audio file", type=['wav', 'mp3', 'flac', 'ogg'])
        if uploaded_file is not None:
            audio_file = uploaded_file
            st.audio(uploaded_file)
            
    with tab2:
        recorded_audio = st.audio_input("Record directly from your microphone:")
        if recorded_audio is not None:
            audio_file = recorded_audio

    st.subheader("2. Speaker Profile (Optional)")
    enrolled_user = st.text_input("Enrolled Username (for 1v1 verification check) :")

    st.markdown("---")
    
    if audio_file is not None and st.button("RUN INFERENCE", type="primary", use_container_width=True):
        with st.spinner("Analyzing spectral signatures and codec artifacts..."):
            t0 = time.time()
            
            # Save bytes to temp file for Torchaudio robust parsing
            tmp_path = save_uploaded_audio(audio_file)
            
            try:
                waveform, spectrogram = audio_pipe.process_file(tmp_path)
                
                # Fetch Speaker profile
                user_emb = None
                if enrolled_user:
                    vp_path = os.path.join(cfg["paths"]["voiceprint_dir"], f"{enrolled_user}.npy")
                    if os.path.exists(vp_path):
                        user_emb = np.load(vp_path)
                    else:
                        st.warning(f"Voiceprint for '{enrolled_user}' not found.")
                
                # Inference
                results = ensemble_engine.evaluate(waveform, spectrogram, enrolled_user_emb=user_emb)
                
                display_results(results)
                st.caption(f"Inference latency: {time.time()-t0:.2f} seconds")
                
            except Exception as e:
                st.error(f"Error during audio processing: {str(e)}")
            finally:
                cleanup_temp_file(tmp_path)

if __name__ == "__main__":
    main()
