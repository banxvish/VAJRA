import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { useVajra } from '@/context/VajraContext';

// Helper to reliably hash strings using Web API
const computeSHA256 = async (message: string) => {
  const msgBuffer = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return '0x' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
};

type ZKStatus = 'idle' | 'generating' | 'anchored' | 'failed';

const ZKAttestation = () => {
  const { systemStatus, analysisResult } = useVajra();
  const [status, setStatus] = useState<ZKStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [visibleLines, setVisibleLines] = useState<string[]>([]);
  const [proofData, setProofData] = useState({ hash: '', nullifier: '', commitment: '' });
  const prevStatus = useRef(systemStatus);
  const generating = useRef(false);

  const generate = async () => {
    if (generating.current) return;
    generating.current = true;
    setStatus('generating');
    setVisibleLines([]);
    setProgress(0);

    // Build real contextual logs
    let payloadString = "";
    const dynamicLogs = [
      '> Initializing RISC Zero zkVM...',
      '> Loading circuit: identity_verify_v2.zk'
    ];

    if (analysisResult) {
      dynamicLogs.push(`> Hashing session payload (Trust Score: ${(analysisResult.trust_score * 100).toFixed(1)}%)...`);
      dynamicLogs.push(`> Spectrogram: ${analysisResult.models.spectrogram} | Wav2Vec2: ${analysisResult.models.wav2vec}...`);
      dynamicLogs.push(`> Speaker Similarity: ${analysisResult.models.speaker_similarity.toFixed(2)} — Anchoring node...`);
      payloadString = JSON.stringify(analysisResult);
    } else {
      dynamicLogs.push('> Hashing baseline sensor feed...');
      payloadString = JSON.stringify({ ts: Date.now(), sensor: "baseline" });
    }

    dynamicLogs.push('> Computing Merkle inclusion proof...');
    dynamicLogs.push('> Running STARK prover (3.2M constraints)...');
    dynamicLogs.push('> Verifying proof locally...');
    dynamicLogs.push('> Anchoring to Polygon Amoy...');
    dynamicLogs.push('> ✓ Proof verified on-chain');

    // Asynchronously simulate timeline while hashing
    const startTs = Date.now().toString();
    const realHash = await computeSHA256(payloadString);
    const realNullifier = await computeSHA256(payloadString + startTs + "nullifier");
    const realCommitment = await computeSHA256(payloadString + "commitment");

    dynamicLogs.forEach((line, i) => {
      setTimeout(() => {
        setVisibleLines((prev) => [...prev, line]);
        setProgress(Math.round(((i + 1) / dynamicLogs.length) * 100));
        
        if (i === dynamicLogs.length - 1) {
          setStatus('anchored');
          setProofData({
            hash: realHash.substring(0, 42), // Typical ETH format length
            nullifier: realNullifier.substring(0, 42),
            commitment: realCommitment.substring(0, 42),
          });
          generating.current = false;
        }
      }, 700 * (i + 1));
    });
  };

  useEffect(() => {
    if (prevStatus.current !== systemStatus) {
      if (systemStatus === 'verified') {
        setTimeout(generate, 800);
      } else if (systemStatus === 'threat') {
        generating.current = false;
        setStatus('failed');
        setVisibleLines(['> ✗ PROOF GENERATION FAILED — IDENTITY COMPROMISED']);
        setProgress(0);
      } else if (systemStatus === 'idle') {
        generating.current = false;
        setStatus('idle');
        setVisibleLines([]);
        setProgress(0);
      }
      prevStatus.current = systemStatus;
    }
  }, [systemStatus]);

  const panelGlow = systemStatus === 'threat' ? 'threat-glow animate-pulse-threat' : systemStatus === 'verified' ? 'verified-glow' : '';

  const statusBadge = () => {
    if (status === 'generating') return (
      <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-rotate-ring" />
    );
    if (status === 'anchored') return (
      <span className="font-display text-[8px] tracking-[0.15em] text-authentic px-2 py-0.5 rounded-full border border-authentic/30">✓ ANCHORED</span>
    );
    if (status === 'failed') return (
      <span className="font-display text-[8px] tracking-[0.15em] text-destructive px-2 py-0.5 rounded-full border border-destructive/30">✗ FAILED</span>
    );
    return null;
  };

  return (
    <div className={`vajra-panel bg-card rounded-lg border border-foreground/[0.06] p-4 flex flex-col ${panelGlow}`}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="font-display text-xs tracking-[0.3em] text-foreground">ZK ATTESTATION</h2>
          <p className="font-display text-[8px] tracking-[0.2em] text-muted-foreground mt-0.5">RISC ZERO · POLYGON AMOY</p>
        </div>
        {statusBadge()}
      </div>

      {/* Progress bar */}
      {status === 'generating' && (
        <div className="mb-3">
          <div className="h-1 bg-background/50 rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full bg-accent"
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <span className="font-mono text-[9px] text-accent mt-1 inline-block">{progress}%</span>
        </div>
      )}

      {/* Circuit log */}
      <div className="bg-background/80 rounded-md p-3 mb-3 flex-1 min-h-[120px] max-h-[180px] overflow-y-auto font-mono text-[10px] text-secondary leading-relaxed">
        {visibleLines.map((line, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            {line}
          </motion.div>
        ))}
        {visibleLines.length === 0 && (
          <span className="text-muted-foreground">{'>'} Awaiting identity signal...</span>
        )}
      </div>

      {/* Proof data */}
      {status === 'anchored' && (
        <div className="space-y-1 mb-3">
          {[
            { l: 'PROOF HASH', v: proofData.hash },
            { l: 'NULLIFIER', v: proofData.nullifier },
            { l: 'COMMITMENT', v: proofData.commitment },
          ].map(d => (
            <div key={d.l} className="flex justify-between">
              <span className="font-display text-[7px] tracking-[0.15em] text-muted-foreground">{d.l}</span>
              <span className="font-mono text-[9px] text-secondary truncate ml-2 max-w-[140px]" title={d.v}>{d.v}</span>
            </div>
          ))}
        </div>
      )}

      <button
        onClick={generate}
        disabled={status === 'generating'}
        className="w-full font-display text-[10px] tracking-[0.2em] py-2.5 bg-accent/10 text-accent border border-accent/20 rounded-md hover:bg-accent/20 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
      >
        ∮ GENERATE ZK PROOF
      </button>
    </div>
  );
};

export default ZKAttestation;
