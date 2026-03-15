# KAVACHA - Frontend Dashboard 🛡️

This is the central operating system and Web3 dashboard for the KAVACHA Zero-Trust Identity Defense Platform. It acts as a real-time command center, connecting WebRTC device streams, Live Machine Learning inferences, and Blockchain Smart Contracts into a single flawless UX.

## 🚀 Technologies Used
* **React 18 & Vite:** High-performance, rapid-rendering UI engine.
* **Framer Motion:** Smooth, hardware-accelerated animations and threat visualizations.
* **Tailwind CSS:** Enterprise-grade, modern utility-first styling.
* **Lucide React:** Beautiful, consistent iconography.
* **WebRTC & Canvas API:** Browser-native camera indexing and frame extractions required for rPPG Liveness validation and deepfake perturbations.
* **Ethers.js (v6):** Web3 RPC binding layer to broadcast live Zero-Knowledge proof hashes directly to the Polygon Amoy blockchain.
* **WebCrypto API:** In-browser deterministic SHA-256 payload hashing (`crypto.subtle`) ensuring data integrity before blockchain commits.

## 📁 Key Components
* `VideoShield.tsx`: Captures camera streams, evaluates Xception spatio-temporal outputs, and manages the rPPG (Remote Photoplethysmography) phase-shift flash sequence.
* `VoiceEngine.tsx`: The audio portal. Connects a live microphone feed to the RAKSHA Ensemble and visualizes Audio Spectrograms and Codec Ringing metrics.
* `ZKAttestation.tsx`: Demonstrates Zero-Knowledge Proof bridging by hashing real-time interaction states.
* `BlockchainLedger.tsx`: Uses Ethers.js to ping the live `KavachaTrustRegistry` Solidity smart contract on Polygon when `SAFE` or `FAKE` events trigger.

## 🛠️ Quick Start
1. Ensure you have **Node 18+** installed.
2. Navigate into this directory (`/frontend`)
3. Install dependencies:
   ```bash
   npm install
   ```
4. Start the Vite development server:
   ```bash
   npm run dev
   ```
5. Open your browser to `http://localhost:5173`.
