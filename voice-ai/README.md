# KAVACHA - Machine Learning Backend 🧠

This is the brain of the KAVACHA Zero-Trust Identity Defense Platform. It is an extremely modular, scalable machine learning microservice built on FastAPI to process incoming Voice and Video feeds in real-time, executing Bayesian Ensembles and Adversarial Model Collapses.

## 🚀 Technologies Used
* **FastAPI:** Lightning-fast, modern Python API framework used to build asynchronous ingestion endpoints.
* **PyTorch:** The underlying neural network tensor framework executing all deep-learning models.
* **OpenCV (`cv2`):** Core computer vision utility stack used to unpack, decode, and transform raw webcam frame blobs.
* **Torchaudio:** High-performance, low-latency audio file processing format extracting features seamlessly.
* **Librosa:** Deep audio DSP stack extracting the Mel-Spectrogram signatures needed to identify neural voice clones (ElevenLabs, Coqui TTS, etc).
* **XceptionNet & EfficientNet:** The pre-trained computer vision architectures used for high-fidelity Spatial/Temporal analysis of human faces and deepfakes.

## 📁 Key Components
* `backend/main.py`: The single entry API server orchestrating all data flow via unified POST endpoints.
* `backend/pipeline.py`: A Bayesian Ensemble logic gate crossing Wav2Vec2 + ECAPA-TDNN scoring, significantly reducing False Positives caused by noise environments.
* `models/voice_model.py`: Calculates probabilities of neural text-to-speech codec "ringing" artifacts.
* `models/video_model.py`: Detects missing facial landmarks, temporal gltches, and structural anomalies in moving video frame blobs using Xception weights.

## 🛠️ Quick Start
1. Ensure you have **Python 3.10+** installed.
2. Navigate into this directory (`/voice-ai`)
3. Create and target a new virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   # OR
   venv\Scripts\activate           # Windows
   ```
4. Install all dependencies from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the FastAPI server using Uvicorn on Port 8000:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
6. The service is now live and waiting for data from the KAVACHA React Dashboard!
