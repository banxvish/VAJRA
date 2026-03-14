"""
KAVACHA Voice Deepfake Detection — Speaker Verification
========================================================
Pretrained ECAPA-TDNN from SpeechBrain for:
  • Speaker embedding extraction (192-dim)
  • Speaker enrollment (save voiceprint)
  • Speaker verification (cosine similarity)

This model is NOT retrained — only used for inference.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio

from training.preprocess import load_audio, pad_or_trim, normalize_waveform
from training.utils import ensure_dirs, setup_logger

logger = setup_logger("speaker_verification")


class SpeakerVerifier:
    """
    Wrapper around the pretrained SpeechBrain ECAPA-TDNN model
    for speaker embedding extraction and verification.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Parameters
        ----------
        cfg : dict
            Full YAML configuration with ``speaker_verification`` section.
        """
        sv_cfg = cfg["speaker_verification"]
        self.model_source = sv_cfg["model_source"]
        self.embedding_dim = sv_cfg["embedding_dim"]
        self.similarity_threshold = sv_cfg["similarity_threshold"]
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.duration = cfg["audio"]["duration"]
        self.target_length = self.sample_rate * self.duration
        self.voiceprint_dir = cfg["paths"]["voiceprint_dir"]

        # Lazy load SpeechBrain model
        self._model = None

    def _load_model(self):
        """Load pretrained ECAPA-TDNN from SpeechBrain (lazy)."""
        if self._model is not None:
            return

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        logger.info(f"Loading ECAPA-TDNN from: {self.model_source}")
        self._model = EncoderClassifier.from_hparams(
            source=self.model_source,
            savedir="pretrained_models/ecapa_tdnn",
            run_opts={"device": "cpu"},
        )
        logger.info("ECAPA-TDNN loaded successfully.")

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract a 192-dimensional speaker embedding from an audio file.

        Parameters
        ----------
        audio_path : str
            Path to audio file.

        Returns
        -------
        embedding : np.ndarray of shape (192,)
        """
        self._load_model()

        waveform, _ = load_audio(audio_path, target_sr=self.sample_rate)
        waveform = pad_or_trim(waveform, target_length=self.target_length)
        waveform = normalize_waveform(waveform)

        # SpeechBrain expects (batch, time)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)  # (time,)
        waveform = waveform.unsqueeze(0)    # (1, time)

        embedding = self._model.encode_batch(waveform)
        embedding = embedding.squeeze().detach().cpu().numpy()
        return embedding

    def enroll_speaker(self, user_id: str, audio_path: str) -> np.ndarray:
        """
        Enroll a speaker by extracting and saving their voiceprint.

        Parameters
        ----------
        user_id : str
            Unique speaker identifier.
        audio_path : str
            Path to enrollment audio.

        Returns
        -------
        embedding : np.ndarray of shape (192,)
        """
        embedding = self.extract_embedding(audio_path)

        ensure_dirs(self.voiceprint_dir)
        save_path = os.path.join(self.voiceprint_dir, f"{user_id}.npy")
        np.save(save_path, embedding)
        logger.info(f"Enrolled speaker '{user_id}' → {save_path}")

        return embedding

    def load_voiceprint(self, user_id: str) -> np.ndarray:
        """Load a previously enrolled voiceprint."""
        vp_path = os.path.join(self.voiceprint_dir, f"{user_id}.npy")
        if not os.path.exists(vp_path):
            raise FileNotFoundError(
                f"No voiceprint found for user '{user_id}' at {vp_path}"
            )
        return np.load(vp_path)

    def verify_speaker(
        self,
        user_id: str,
        audio_path: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Verify whether the speaker in *audio_path* matches the enrolled
        voiceprint of *user_id*.

        Parameters
        ----------
        user_id : str
            Enrolled speaker identifier.
        audio_path : str
            Path to verification audio.
        threshold : float, optional
            Cosine similarity threshold. Defaults to config value.

        Returns
        -------
        is_match : bool
        similarity : float  (cosine similarity in [-1, 1])
        """
        if threshold is None:
            threshold = self.similarity_threshold

        enrolled = self.load_voiceprint(user_id)
        test_emb = self.extract_embedding(audio_path)

        similarity = self._cosine_similarity(enrolled, test_emb)
        is_match = similarity >= threshold

        logger.info(
            f"Verification for '{user_id}': similarity={similarity:.4f}, "
            f"threshold={threshold:.4f}, match={is_match}"
        )
        return is_match, similarity

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def batch_extract_embeddings(
        self, audio_paths: list
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of audio files.

        Returns
        -------
        embeddings : np.ndarray of shape (N, 192)
        """
        embeddings = []
        for path in audio_paths:
            emb = self.extract_embedding(path)
            embeddings.append(emb)
        return np.stack(embeddings)
