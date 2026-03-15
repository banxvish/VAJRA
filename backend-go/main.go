package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

type VerificationPayload struct {
	ProofBytes         string   `json:"proof_bytes"`
	PublicInputs       []string `json:"public_inputs"`
	TransactionPayload string   `json:"transaction_payload"`
}

type VerificationResponse struct {
	Verified           bool   `json:"verified"`
	ProofHash          string `json:"proof_hash"`
	BlockchainAnchor   string `json:"blockchain_anchor"`
	VerificationTimeMs int    `json:"verification_time_ms"`
	CertificateQrURL   string `json:"certificate_qr_url"`
}

func verifyZKProof(w http.ResponseWriter, r *http.Request) {
	var payload VerificationPayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Simulated RISC Zero verification check:
	// In production, this would call out to `bonsai` or perform a local `STARK` verification
	startTime := time.Now()
	time.Sleep(120 * time.Millisecond) // Simulating math operations
	durationMs := int(time.Since(startTime).Milliseconds())

	resp := VerificationResponse{
		Verified:           true,
		ProofHash:          "0x" + payload.ProofBytes[:min(20, len(payload.ProofBytes))] + "...ZK",
		BlockchainAnchor:   "0xPendingAmoyTxHash",
		VerificationTimeMs: durationMs,
		CertificateQrURL:   "ipfs://QmCertificate...",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func wsHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WS upgrade failed:", err)
		return
	}
	defer conn.Close()

	for {
		messageType, message, err := conn.ReadMessage()
		if err != nil {
			log.Println("WS Read failed:", err)
			break
		}
		
		fmt.Printf("Received %d bytes of audio chunk via WS\n", len(message))
		
		// Echo for now, simulate real-time scoring
		scores := map[string]interface{}{
			"chunk_id": time.Now().UnixNano(),
			"score": 92.5,
			"verdict": "AUTHENTIC",
		}
		
		responseBytes, _ := json.Marshal(scores)
		err = conn.WriteMessage(messageType, responseBytes)
		if err != nil {
			log.Println("WS Write failed:", err)
			break
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	http.HandleFunc("/api/v1/zk/verify-proof", verifyZKProof)
	http.HandleFunc("/ws/voice/stream", wsHandler)

	fmt.Println("KAVACHA Go Microservice running on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
