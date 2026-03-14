# 🛡️ KAVACHA AI — Frontend Dashboard

This represents the React Frontend for the **KAVACHA AI Voice Defense Engine**, an enterprise-grade solution that analyzes audio to detect deepfakes, synthesized codec artifacts, and speaker impersonation.

The frontend is designed to interact simultaneously with the high-performance Python FastAPI backend, delivering sub-second, real-time trust analysis directly to the user.

## ✨ Core Features

- **Real-Time Result Dashboard:** A rich user interface that visualizes deepfake evaluation status instantly.
- **Audio Uploads:** Supports drag-and-drop uploads of specific audio blocks (`.wav`, `.mp3`, etc.).
- **Live Microphone Recording:** Allows intercepting speech straight from the browser microphone to query the server live.
- **Speaker Profiling:** Input generic usernames to instruct the backend to cross-reference identity embeddings automatically.

## ⚖️ Trust Score Weightage (Backend Linked)

The UI visualizes an **Ensemble Trust Score**, which evaluates outputs across exactly four independent Neural Networks executed dynamically on the backend. This is aggregated via the following configured weightage:

- **40% (0.40) — Spectrogram Analysis:** 2D spectral image analysis evaluated by ImageNet-pretrained `EfficientNet-B0`.
- **40% (0.40) — Wav2Vec2 Semantic Deepfake Detection:** Evaluates contextual voice properties sequentially using `facebook/wav2vec2-base`.
- **15% (0.15) — Codec Artifact Prediction:** Detects specific AI generator artifacts (ENCODEC/SOUNDSTREAM) using a custom `1D CNN`.
- **5% (0.05) — Speaker Verification Score:** Analyzes voiceprint similarity derived precisely from SpeechBrain's `ECAPA-TDNN`.

Based on the aggregated trust score, the UI will dynamically render the safety status:
- 🟢 **SAFE**
- 🟡 **SUSPICIOUS**
- 🔴 **FAKE**

## 🔧 Technologies Used

This interactive UI utilizes lightweight web standards heavily optimized for speed:
- **React.js** (Frontend Framework)
- **Vite** (Build Tool)
- **TypeScript** (Static Typing)
- **Tailwind CSS** (Styling)
- **shadcn-ui** (Accessible UI Components)

## 📡 Backend API Integration

The application natively talks to the KAVACHA Python logic via REST interface endpoints exposed over `CORS`:

1. `POST /analyze_audio`: Primary multimedia submission endpoint returning the aggregate Trust Score arrays.
2. `POST /enroll_speaker` / `POST /verify_speaker`: Secondary Identity embedding utilities handled natively.

**(Make sure the Python `FastAPI` server is actively running on `http://localhost:8000` to ensure proper linkage).**

## 🚀 Running the Frontend Development Server

If you want to spin up the UI locally, follow these steps:

```sh
# Step 1: Navigate to the project directory.
cd c:\Users\bavis\OneDrive\Documents\VAJRA\frontend

# Step 2: Install dependencies (requires Node.js & npm).
npm install

# Step 3: Start the Vite developement server.
npm run dev
```

Your React app will generally be exposed on `http://localhost:8080` or `http://localhost:5173`.
