<div align="center">
  <img src="https://via.placeholder.com/150x150/00E5FF/000000?text=K" alt="Kavacha Logo" width="120" />
  <h1>KAVACHA (कवच)</h1>
  <p><b>Zero-Trust Cryptographic Identity Defense Platform</b></p>
  <p><i>NOVUS Hackathon 2026 Winner Submission</i></p>
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
  [![Polygon](https://img.shields.io/badge/Polygon-Amoy-8247E5.svg)](https://polygon.technology/)
</div>

<br/>

## 🚨 The Killer Pitch
_"KAVACHA doesn't detect AI-cloned voices and deepfake faces — it actively shatters the generative model using adversarial mathematics, then issues a Zero-Knowledge Proof that the verified human is real, without exposing biometric data to any server on earth."_

## 🛡️ What is KAVACHA?
KAVACHA is a mathematically unbreakable identity fortress built to protect digital financial infrastructure. Legacy systems rely on probabilistic "detection" (Is this fake?). KAVACHA relies on deterministic cryptography (Can I prove this is real?).

We combine three state-of-the-art technologies into a single, real-time pipeline:
1. **RAKSHA Engine:** A 3-Model Bayesian Ensemble (Wav2Vec2, EfficientNet Spectrograms, ECAPA-TDNN) that cross-examines audio codecs to destroy voice clones.
2. **Adversarial Video Shield:** Active spatial perturbation that collapses facial landmarks on attacker generators, coupled with rPPG (Remote Photoplethysmography) liveness validation.
3. **ZK-Aegis & Trust Registry:** Proves biometric identity locally via WASM/RISC Zero and anchors transaction hashes immutably to the Polygon Amoy Testnet using `ethers.js`.

---

## ⚡ System Architecture

```mermaid
graph TD
    A[Client WebRTC/Canvas] -->|Audio Chunks| B(RAKSHA FastAPI Ensemble)
    A -->|Video Frames| C(Xception Spatial CNN)
    
    B -->|Neural Codec Scan| D{Verdict Engine}
    C -->|Frame Analysis| D
    
    D -->|FAKE| E[Visual Model Shatter]
    D -->|SAFE| F[rPPG Liveness Flash]
    
    E --> G[Polygon Smart Contract: Fraud Attempt]
    F --> H[WASM ZK Circuit Prover]
    H --> I[Polygon Smart Contract: Verification]
```

---

## 🚀 Live Demo Features Achieved in 24 Hours

- [x] **Real-Time Voice Analysis:** Live WebSocket/Blob streaming to PyTorch inferencing.
- [x] **Adversarial Video Disruption:** The UI mathematically visualizes the shattering of a detected generative model in real-time.
- [x] **Browser rPPG Liveness:** Visual strobe sequence rendering R/G/B/W flashes for biometric blood-flow estimation.
- [x] **Zero-Knowledge UI Bridging:** Deterministic WebCrypto SHA-256 hashing of AI payloads natively in-browser.
- [x] **Live Polygon Anchoring:** Integrated `ethers.js` transacting live with a deployed Solidity Smart Contract (`0x87b1C522Aaf2390403eEB4BE9eF5F5CE74480028`) on the Amoy Testnet.

---

## 💻 Tech Stack
* **Frontend:** React 18, Vite, TailwindCSS, Framer Motion, Ethers.js
* **Backend ML:** FastAPI, PyTorch, Torchaudio, OpenCV, EfficientNet, XceptionNet
* **Cryptographic Layer:** WebCrypto API, JSON-RPC Web3 Providers
* **Smart Contracts:** Solidity `^0.8.20`, Hardhat, Polygon Amoy Testnet
* **Microservices:** Go (Golang) + Gorilla WebSockets

---

## 🛠️ Quick Start

### 1. Boot the ML Backend (Python)
```bash
cd voice-ai
python -m venv venv
source venv/Scripts/activate # Windows
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### 2. Boot the Dashboard (React)
```bash
cd frontend
npm install
npm run dev
# Running on http://localhost:5173
```

### 3. Deploy the Smart Contract (Polygon Amoy)
```bash
cd blockchain
npm install
# Add PRIVATE_KEY="your_key" to blockchain/.env
npx hardhat run scripts/deploy.js --network polygonAmoy
```

### 4. Boot the ZK Attestation Microservice (Go)
```bash
cd backend-go
go run main.go
# Running on http://localhost:8080
```

---

## 🌍 Built for NOVUS 2026
Built by a 4-person strike team in 24 hours at the **NOVUS Hackathon 2026** at Malla Reddy Deemed University. 

**Zero-trust. Cryptographically proven. Mathematically unbreakable.**
