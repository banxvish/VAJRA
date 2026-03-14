"""
KAVACHA AI Voice Defense Engine — API Schemas
===============================================
Pydantic schemas for the FastAPI backend.
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel


class ModelsResult(BaseModel):
    spectrogram: str
    wav2vec: str
    codec: str
    speaker_similarity: float


class AnalyzeAudioResponse(BaseModel):
    trust_score: float
    status: str
    models: ModelsResult


class EnrollSpeakerResponse(BaseModel):
    message: str
    user_id: str


class VerifySpeakerResponse(BaseModel):
    similarity: float
    match: bool
