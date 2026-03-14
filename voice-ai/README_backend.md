# KAVACHA AI Backend

The KAVACHA AI Backend is a high-performance Python inference engine built with FastAPI and PyTorch. It is responsible for orchestrating the audio processing pipeline and routing the data through our 4 independent ML models to detect audio deepfakes and verify speaker identity.

## Environment Setup
It is highly recommended to run this backend in a Python virtual environment to prevent dependency conflicts. The environment should use Python 3.9+.

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

## Model Training Instructions
If you wish to retrain any of the component models, use the centralized master script:

```bash
# Train all models sequentially
python training/train_all.py

# To skip certain models, you can use flags (if configured):
python training/train_all.py --skip_codec
```

All training hyperparameters (such as epochs, dataset paths, learning rates) are strictly defined in `configs/training.yaml`.

## API Endpoints
The FastAPI server exposes 3 primary REST endpoints.

### `POST /analyze_audio`
Receives an audio file upload, extracts features, processes it through the entire Model Ensemble (Spectrogram, Wav2Vec2, Codec), and returns a trust score evaluating whether the audio is `SAFE` or a `FAKE`.
- **Form-Data Parameter:** `audio` -> The `.wav` or standard audio file blob.
- **Returns:** JSON containing `"trust_score"`, `"status"`, and individual model votes.

### `POST /enroll_speaker`
Generates a 192-dimensional ECAPA-TDNN voiceprint mathematically representing an identity.
- **Form-Data Parameter:** `user_id` -> A unique string identifier.
- **Form-Data Parameter:** `audio` -> The clean base audio file of the speaker.
- **Returns:** JSON containing an enrollment success message.

### `POST /verify_speaker`
Compares incoming audio against a pre-enrolled identity using cosine similarity.
- **Form-Data Parameter:** `user_id` -> The enrolled user identifier.
- **Form-Data Parameter:** `audio` -> The audio trace to verify.
- **Returns:** JSON containing `"similarity"` score and a `"match"` boolean.

## Server Startup
To start the backend for active API processing and API routing:
```bash
uvicorn backend.main:app --reload
```
The server will boot up the models dynamically into memory. Once finished, you will see `Application startup complete`. You can visit `http://127.0.0.1:8000/docs` to interact with the interactive Swagger dashboard.
