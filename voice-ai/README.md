# 🛡️ KAVACHA AI Voice Defense Engine

**KAVACHA** is a production-grade, real-time AI identity defense system designed to protect against voice deepfakes, neural codec artifacts, and speaker impersonation. 

This repository contains the complete modular Machine Learning training pipeline, featuring advanced self-supervised audio models, dynamic ensemble trust scoring, and an interactive hackathon dashboard.

---

## 🚀 Features

The defense engine relies on a **4-Model Distributed Ensemble** that combines different acoustic and semantic perspectives to produce a final Trust Score:

1. **Spectrogram Deepfake Detector**
   - **Architecture:** `EfficientNet-B0` (Pretrained on ImageNet).
   - **Input:** 128×128 Mel Spectrogram Images.
   - **Purpose:** Detects 2D spectral artifacts left by generative models.
2. **Wav2Vec2 Self-Supervised Detector**
   - **Architecture:** `facebook/wav2vec2-base` (HuggingFace Transformers).
   - **Input:** Raw 16kHz Waveform (2 seconds).
   - **Purpose:** Captures semantic voice inconsistencies and synthesis artifacts by pooling hidden states.
3. **Codec Artifact Detector**
   - **Architecture:** Lightweight Custom `1D CNN`.
   - **Input:** Raw Waveform (32,000 samples).
   - **Purpose:** Sub-millisecond neural codec matching (`HUMAN`, `ENCODEC`, `SOUNDSTREAM`).
4. **Speaker Verification Engine**
   - **Architecture:** `ECAPA-TDNN` (SpeechBrain).
   - **Input:** Raw Waveform.
   - **Purpose:** Extracts 192-dimensional speaker voiceprints and evaluates cosine similarity. (Not retrained).

### Pipeline Capabilities
- **Modular Design:** Strictly separated packages for datasets, preprocessing, models, training, evaluation, algorithms, and exports.
- **Automated Datasets:** Built-in downloader for ASVspoof 2019 LA and LibriSpeech, complete with automatic dummy-data fallback generation if offline.
- **GPU Optimization:** Mixed-Precision Training (AMP), PyTorch `GradScaler`, and gradient clipping.
- **Data Augmentation:** Gaussian noise, pitch shifting, time-stretching, background noise, and SNR-controlled room reverberation.
- **Experiment Tracking:** TensorBoard integration + automated JSON metadata extraction for full reproducibility.
- **Deployment & Export:** PyTorch `.pt` dictionary states and dynamically-batched automated `ONNX` exports.
- **Zero Hardcoding:** 100% configuration-driven via `configs/training.yaml`.

---

## 📂 Project Architecture

```text
voice-ai/
├── configs/
│   └── training.yaml              # Central configuration (hyperparameters, paths, etc.)
├── datasets/
│   ├── download_datasets.py       # Automated dataset fetching and extraction
│   └── dataset_builder.py         # PyTorch DataLoaders with WeightedRandomSampler
├── preprocessing/
│   ├── audio_processor.py         # Resampling, mono conversion, trimming, normalization
│   ├── spectrogram_generator.py   # Log-mel spectrogram generation
│   └── augmentations.py           # Pitch shift, time stretch, noise, reverb
├── models/
│   ├── spectrogram_model.py       # EfficientNet-B0 architecture
│   ├── wav2vec_detector.py        # HuggingFace Wav2Vec2 architecture
│   ├── codec_detector.py          # 1D CNN architecture
│   └── speaker_verification.py    # SpeechBrain ECAPA-TDNN wrapper
├── training/
│   ├── trainer.py                 # BaseTrainer class with AMP, early stopping, TensorBoard
│   ├── train_spectrogram.py       # Two-stage transfer learning for Spectrogram
│   ├── train_wav2vec.py           # Two-stage fine-tuning for Wav2Vec2
│   ├── train_codec.py             # Single-stage training for Codec detector
│   └── train_all.py               # Master orchestration script
├── evaluation/
│   ├── metrics.py                 # F1, Accuracy, ROC-AUC, Equal Error Rate (EER)
│   └── evaluate_models.py         # Evaluation runner
├── experiments/
│   └── experiment_logger.py       # JSON metadata and tracking
├── export/
│   └── export_models.py           # .pt and .onnx model exporters
├── inference/
│   ├── model_loader.py            # Automated device placement and batch loader
│   ├── audio_pipeline.py          # Reproduces training preprocessing during inference
│   └── ensemble_engine.py         # The trust score aggregation logic
├── utils/
│   ├── seed.py, device.py, checkpoint.py # Core ML utilities
│   ├── audio_utils.py             # Temp file IO wrappers for streaming
│   └── visualization.py           # Console UI terminal formatting
├── dashboard/
│   └── app.py                     # Streamlit web UI
├── demo/
│   └── run_demo.py                # Standalone CLI deployment test
└── requirements.txt               # Python dependencies
```

---

## 🛠️ Installation

```powershell
# 1. Clone the repository and navigate to the AI directory
git clone https://github.com/your-org/KAVACHA.git
cd KAVACHA/voice-ai

# 2. Create and activate a Virtual Environment
python -m venv venv
venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 🧠 Master Training Pipeline

The entire pipeline (Dataset Download ➔ Preprocessing ➔ Modality Trainings ➔ Evaluation ➔ ONNX Export) can be triggered with a single command. 

```powershell
# Run the complete pipeline
python training/train_all.py

# Run training but skip specific models
python training/train_all.py --skip_codec --skip_wav2vec

# Only run evaluation on previously trained models
python training/train_all.py --eval_only
```

Hyperparameters are fully customizable inside `configs/training.yaml`.

---

## 🌐 Hackathon Streamlit Dashboard

The web dashboard provides a clean UI where judges can record audio directly from their browser or upload pre-recorded test files to see the Ensemble analyze the data in `< 2 seconds`. 

```powershell
streamlit run dashboard/app.py
```

Features:
- **Audio Input Options:** Upload audio files (`.wav`, `.mp3`) or record a live 2s clip directly from the browser window.
- **Speaker Profile Loading:** Check similarity against pre-enrolled identity profiles dynamically.
- **Trust Score Progress Bar:** Instantly visualizes the unified Safe/Suspicious/Fake classification mathematically aggregating the system.

---

## 🎤 Command-Line Inference Demo

If you prefer testing heavily in the system terminal without rendering a full webpage, use the deployment CLI.

```powershell
# Evaluate an existing audio file on disk
python demo/run_demo.py --file path/to/suspicious_audio.wav

# Verify identity against an enrolled user profile
python demo/run_demo.py --file my_voice.wav --enrolled_user "ceo_profile"
```

**Example CLI Output:**
```text
KAVACHA Analysis Result
=======================

Trust Score: 0.81
Status:      SAFE

Model Votes
Spectrogram:   REAL
Wav2Vec2:      REAL
Codec:         HUMAN
Speaker Match: HIGH (0.83)
```

---

## ⚙️ Configuration (`training.yaml`)

Edit the core config to control training dynamics. Example subsets:

```yaml
audio:
  sample_rate: 16000
  duration: 2

spectrogram:
  n_mels: 128
  image_size: 128

wav2vec:
  model_name: "facebook/wav2vec2-base"
  epochs_stage1: 10
  epochs_stage2: 5

ensemble:
  spectrogram_weight: 0.4
  wav2vec_weight: 0.4
  codec_weight: 0.15
  speaker_weight: 0.05
```
