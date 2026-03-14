"""
KAVACHA AI Voice Defense Engine — Speaker Verification
=======================================================
Pretrained ECAPA-TDNN from SpeechBrain.
  • Extract 192-dim speaker embeddings
  • Enroll speakers (save voiceprint as .npy)
  • Verify via cosine similarity

This model is NOT retrained.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

from preprocessing.audio_processor import load_audio, pad_or_trim, normalize_waveform


class SpeakerVerifier:
    """ECAPA-TDNN speaker verification wrapper."""

    def __init__(self, cfg: Dict[str, Any]):
        sv = cfg["speaker_verification"]
        self.source = sv["model_source"]
        self.emb_dim = sv["embedding_dim"]
        self.threshold = sv["similarity_threshold"]
        self.sr = cfg["audio"]["sample_rate"]
        self.target_len = cfg["audio"]["target_length"]
        self.vp_dir = cfg["paths"]["voiceprint_dir"]
        self._model = None

    def _load(self):
        import huggingface_hub
        _old_download = huggingface_hub.hf_hub_download
        def _new_download(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            kwargs["local_dir_use_symlinks"] = False
            return _old_download(*args, **kwargs)
        huggingface_hub.hf_hub_download = _new_download

        import shutil
        import os
        _old_symlink = getattr(os, "symlink", None)
        def _new_symlink(src, dst, *args, **kwargs):
            shutil.copyfile(src, dst)
        os.symlink = _new_symlink

        from speechbrain.inference.speaker import EncoderClassifier
        savedir = "pretrained_models/ecapa_tdnn/"
        os.makedirs(savedir, exist_ok=True)
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir
        )
        
        if _old_symlink:
            os.symlink = _old_symlink
        huggingface_hub.hf_hub_download = _old_download

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract 192-dim embedding from audio file."""
        import soundfile as sf
        import numpy as np
        try:
            data, fs = sf.read(audio_path)
        except Exception:
            import librosa
            data, fs = librosa.load(audio_path, sr=None, mono=False)
            if data.ndim == 2:
                data = data.T
                
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        signal = torch.tensor(data.astype(np.float32)).T
        embeddings = self._model.encode_batch(signal)
        return embeddings.squeeze().detach().cpu().numpy()

    def extract_embedding_from_tensor(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract embedding from a preprocessed waveform tensor."""
        embeddings = self._model.encode_batch(waveform)
        return embeddings.squeeze().detach().cpu().numpy()

    def enroll(self, user_id: str, audio_path: str) -> np.ndarray:
        """Enroll speaker and save voiceprint."""
        emb = self.extract_embedding(audio_path)
        Path(self.vp_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(self.vp_dir, f"{user_id}.npy"), emb)
        return emb

    def verify(self, user_id: str, audio_path: str,
               threshold: Optional[float] = None) -> Tuple[bool, float]:
        """Verify a speaker against enrolled voiceprint."""
        thr = threshold or self.threshold
        enrolled = np.load(os.path.join(self.vp_dir, f"{user_id}.npy"))
        test = self.extract_embedding(audio_path)
        sim = float(np.dot(enrolled, test) /
                     (np.linalg.norm(enrolled) * np.linalg.norm(test) + 1e-8))
        return sim >= thr, sim

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
