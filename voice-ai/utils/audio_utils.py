"""
KAVACHA AI Voice Defense Engine — Audio Utilities
===================================================
Helpers for managing real-time audio from Streamlit bytes or files.
"""

import os
import tempfile
from typing import BinaryIO

import torchaudio
import torch

def save_uploaded_audio(upload: BinaryIO, suffix: str = ".wav") -> str:
    """
    Saves a Streamlit UploadedFile or BytesIO object to a temporary file 
    for processing by torchaudio.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(upload.read())
        return f.name

def cleanup_temp_file(filepath: str) -> None:
    """Deletes temporary audio files."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass
