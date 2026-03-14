"""
KAVACHA AI Voice Defense Engine — FastAPI REST Backend
========================================================
Production Inference service acting as the middleware bridging 
the PyTorch ensemble models with the React Frontend Dashboard.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add project root to sys.path so utils imports work out of the box
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
try:
    torchaudio.set_audio_backend('soundfile')
except Exception:
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = True

import os
import shutil
_old_symlink = getattr(os, "symlink", None)
def _new_symlink(src, dst, *args, **kwargs):
    try:
        shutil.copyfile(src, dst)
    except Exception:
        pass
os.symlink = _new_symlink

import huggingface_hub
_old_download = huggingface_hub.hf_hub_download
def _new_download(*args, **kwargs):
    kwargs.pop("use_auth_token", None)
    return _old_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _new_download

from utils.checkpoint import load_config
from backend.schemas import AnalyzeAudioResponse, EnrollSpeakerResponse, VerifySpeakerResponse
from backend.model_loader import APIModelLoader
from backend.audio_processor import APIAudioProcessor
from backend.inference_engine import APIInferenceEngine

# Configuration Setup
try:
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs/training.yaml"))
except Exception as e:
    raise RuntimeError(f"Unable to locate configs/training.yaml format. {e}")

# Global references (assigned uniquely in lifespan)
audio_pipe = None
inference_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook — eagerly allocates Neural Networks onto device at boot."""
    global audio_pipe, inference_engine
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"KAVACHA Backend starting. Binding models to {device}...")
    
    loader = APIModelLoader(cfg, model_dir=os.path.join(PROJECT_ROOT, "models"), device=device)
    loader.initialize()
    
    audio_pipe = APIAudioProcessor(cfg)
    inference_engine = APIInferenceEngine(loader, cfg)
    
    yield
    logging.info("KAVACHA Backend shutting down safely.")

# App Definition
app = FastAPI(
    title="KAVACHA AI Defense Engine API",
    description="High-performance Audio Deepfake inference REST endpoints.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable permissive CORS for React dashboard accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for strict production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_audio", response_model=AnalyzeAudioResponse)
async def analyze_audio(audio: UploadFile = File(...)):
    """
    Core Pipeline: Receives arbitrary audio blobs form-data matching them
    against the ensemble (Spectrogram, Wav2Vec2, Codec) generating a score.
    """
    if audio_pipe is None or inference_engine is None:
        raise HTTPException(status_code=503, detail="Models initializing. Try again momentarily.")
    
    try:
        content = await audio.read()
        waveform, spectrogram = audio_pipe.process_from_bytes(content)
        result = inference_engine.evaluate(waveform, spectrogram, enrolled_emb=None)
        return result
        
    except Exception as e:
        logging.error(f"Prediction Pipeline Failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/enroll_speaker", response_model=EnrollSpeakerResponse)
async def enroll_speaker(user_id: str = Form(...), audio: UploadFile = File(...)):
    """
    Saves an ECAPA-TDNN embedding for a given raw username securely as `.npy`.
    """
    try:
        vp_dir = os.path.join(PROJECT_ROOT, cfg["paths"]["voiceprint_dir"])
        os.makedirs(vp_dir, exist_ok=True)
        
        content = await audio.read()
        waveform, _ = audio_pipe.process_from_bytes(content)
        
        sv = inference_engine.models.speaker_verifier
        emb = sv.extract_embedding_from_tensor(waveform.cpu())
        
        vp_path = os.path.join(vp_dir, f"{user_id}.npy")
        np.save(vp_path, emb)
        
        return EnrollSpeakerResponse(message="Profile enrolled successfully.", user_id=user_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify_speaker", response_model=VerifySpeakerResponse)
async def verify_speaker(user_id: str = Form(...), audio: UploadFile = File(...)):
    """
    Validates generic audio bytes 1v1 against an enrolled user's generated profile.
    """
    try:
        vp_path = os.path.join(PROJECT_ROOT, cfg["paths"]["voiceprint_dir"], f"{user_id}.npy")
        if not os.path.exists(vp_path):
            raise ValueError(f"Enrolled profile {user_id} not found.")

        enrolled_emb = np.load(vp_path)
        
        content = await audio.read()
        waveform, _ = audio_pipe.process_from_bytes(content)
        
        sv = inference_engine.models.speaker_verifier
        test_emb = sv.extract_embedding_from_tensor(waveform.cpu())
        
        sim = float(sv.cosine_similarity(enrolled_emb, test_emb))
        threshold = cfg["speaker_verification"]["similarity_threshold"]

        return VerifySpeakerResponse(similarity=sim, match=(sim >= threshold))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
