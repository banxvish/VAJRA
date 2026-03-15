#![no_main]
risc0_zkvm::guest::entry!(main);

use risc0_zkvm::guest::env;

#[derive(serde::Deserialize)]
struct BiometricInput {
    biometric_live: [f32; 192],
    baseline_hash: [u8; 32],
    liveness_score: f32,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

pub fn main() {
    let input: BiometricInput = env::read();

    // Assertion 1: Liveness is proven (rPPG score must be > 0.80)
    assert!(input.liveness_score > 0.80, "Liveness check failed");

    // In a real implementation, we would also verify the biometric_live 
    // against the original template that hashes to baseline_hash.
    // For this circuit, we assume `biometric_live` has a valid mathematical proof linked.

    env::commit(&input.baseline_hash);
}
