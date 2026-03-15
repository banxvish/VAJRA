"""
KAVACHA AI Voice Defense Engine — Wav2Vec2 Dataset Loader
=========================================================
Loads audio datasets for deepfake detection, specifically adapted
for HuggingFace Wav2Vec2. Includes preprocessing:
 - Resampling to 16kHz
 - Normalizing waveform
 - Chunking/Pad-Trimming to exactly 2 seconds
 - Passing to Wav2Vec2Processor
"""

import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor
import random


def pad_or_trim(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or trim waveform to an exact length (1D)."""
    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)  # Make it 1D
        
    current_len = waveform.shape[0]
    if current_len > target_len:
        # Trim from a random start if training, or just center/start
        # Here we just take the first target_len samples for simplicity
        waveform = waveform[:target_len]
    elif current_len < target_len:
        # Pad with zeros
        pad_len = target_len - current_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    return waveform


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-variance normalization per waveform."""
    mean = waveform.mean()
    std = waveform.std()
    if std > 1e-6:
        return (waveform - mean) / std
    return waveform - mean


class ASVSpoofWav2VecDataset(Dataset):
    """
    Standard PyTorch dataset for Wav2Vec2 deepfake classification.
    Expects a root directory with 'real' and 'fake' subfolders,
    or a metadata file (using folders for simplicity here).
    """
    def __init__(self, data_dir: str, processor: Wav2Vec2FeatureExtractor, target_sr: int = 16000, duration: float = 2.0):
        self.data_dir = data_dir
        self.processor = processor
        self.target_sr = target_sr
        self.target_length = int(target_sr * duration)  # e.g., 32000 samples for 2s
        
        self.samples = []
        
        # Load REAL (Class 0)
        real_files = glob.glob(os.path.join(data_dir, "real", "*.*"))
        for f in real_files:
            if f.endswith((".wav", ".flac", ".mp3", ".ogg")):
                self.samples.append((f, 0))
                
        # Load FAKE (Class 1)
        fake_files = glob.glob(os.path.join(data_dir, "fake", "*.*"))
        for f in fake_files:
            if f.endswith((".wav", ".flac", ".mp3", ".ogg")):
                self.samples.append((f, 1))

        # Shuffle deterministically just in case
        self.samples.sort() 
        random.seed(42)
        random.shuffle(self.samples)

        print(f"Loaded dataset from {data_dir} — {len(self.samples)} files found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # 1. Load Audio Robustly (bypassing torchaudio codec issues)
        import soundfile as sf
        import numpy as np
        
        try:
            audio_data, sr = sf.read(file_path)
            # Make sure it's 2D for processing logic: [frames, channels]
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)
        except Exception:
            import librosa
            audio_data, sr = librosa.load(file_path, sr=None, mono=False)
            if audio_data.ndim == 2:
                audio_data = audio_data.T
            else:
                audio_data = audio_data.reshape(-1, 1)
                
        # Convert to tensor [channels, frames]
        waveform = torch.tensor(audio_data.astype(np.float32)).T
        
        # 2. Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 3. Resample to 16kHz
        if sr != self.target_sr:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            
        waveform = waveform.squeeze() # (L,)
        
        # 4. Normalize and Trim/Pad to 2 seconds
        waveform = normalize_audio(waveform)
        waveform = pad_or_trim(waveform, self.target_length)
        
        # 5. Model Processor
        # Wav2Vec2Processor expects an array/list of floats.
        wav_np = waveform.numpy()
        inputs = self.processor(
            wav_np,
            sampling_rate=self.target_sr,
            padding=True,
            return_tensors="pt"
        )
        # Processor returns (1, L) tensor, we squeeze to (L,)
        input_values = inputs.input_values.squeeze(0)
        
        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }


def get_wav2vec_dataloaders(train_dir: str, val_dir: str, processor: Wav2Vec2FeatureExtractor, batch_size: int = 8):
    """Factory builder for data loaders."""
    train_ds = ASVSpoofWav2VecDataset(train_dir, processor)
    val_ds = ASVSpoofWav2VecDataset(val_dir, processor)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader
