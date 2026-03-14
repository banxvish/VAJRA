import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useVajra } from '@/context/VajraContext';

type TxType = 'PROOF_ANCHOR' | 'IDENTITY_REGISTER' | 'REVOCATION';

interface Tx {
  id: string;
  type: TxType;
  hash: string;
  block: number;
  time: string;
  confirmed: boolean;
}

const typeColors: Record<TxType, string> = {
  PROOF_ANCHOR: '#00E5FF',
  IDENTITY_REGISTER: '#F5A623',
  REVOCATION: '#FF3B5C',
};

const randomHex = () =>
  '0x' + Array.from({ length: 8 }, () => Math.floor(Math.random() * 16).toString(16)).join('') + '...' +
  Array.from({ length: 4 }, () => Math.floor(Math.random() * 16).toString(16)).join('');

const now = () => new Date().toLocaleTimeString('en-US', { hour12: false });

let blockHeight = 48291053;

const BlockchainLedger = () => {
  const { systemStatus } = useVajra();
  const [txs, setTxs] = useState<Tx[]>([]);

  const addTx = (type?: TxType) => {
    const types: TxType[] = ['PROOF_ANCHOR', 'IDENTITY_REGISTER', 'REVOCATION'];
    blockHeight += Math.floor(Math.random() * 3 + 1);
    const tx: Tx = {
      id: Math.random().toString(36).slice(2),
      type: type || types[Math.floor(Math.random() * types.length)],
      hash: randomHex(),
      block: blockHeight,
      time: now(),
      confirmed: Math.random() > 0.2,
    };
    setTxs((prev) => [tx, ...prev].slice(0, 20));
  };

  // Auto-add every 8s
  useEffect(() => {
    const t = setInterval(() => addTx(), 8000);
    return () => clearInterval(t);
  }, []);

  // Listen for ZK proof anchor
  useEffect(() => {
    if (systemStatus === 'verified') {
      setTimeout(() => {
        blockHeight += 1;
        const newTx: Tx = {
          id: Math.random().toString(36).slice(2),
          type: 'PROOF_ANCHOR' as TxType,
          hash: randomHex(),
          block: blockHeight,
          time: now(),
          confirmed: true,
        };
        setTxs((prev) => [newTx, ...prev].slice(0, 20));
      }, 6000); // after ZK proof completes
    }
  }, [systemStatus]);

  const panelGlow = systemStatus === 'threat' ? 'threat-glow animate-pulse-threat' : systemStatus === 'verified' ? 'verified-glow' : '';

  return (
    <div className={`vajra-panel bg-card rounded-lg border border-foreground/[0.06] p-4 flex flex-col flex-1 ${panelGlow}`}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="font-display text-xs tracking-[0.3em] text-foreground">TRUST REGISTRY</h2>
          <p className="font-display text-[8px] tracking-[0.2em] text-muted-foreground mt-0.5">POLYGON AMOY TESTNET</p>
        </div>
        <span className="font-mono text-[9px] text-secondary">BLK {blockHeight}</span>
      </div>

      {/* Transaction list */}
      <div className="flex-1 min-h-[120px] max-h-[200px] overflow-y-auto space-y-1 mb-3">
        {txs.length === 0 && (
          <span className="font-display text-[9px] text-muted-foreground tracking-[0.15em]">AWAITING TRANSACTIONS...</span>
        )}
        {txs.map((tx) => (
          <motion.div
            key={tx.id}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.2 }}
            className="flex items-center gap-2 py-1.5 border-b border-foreground/[0.03]"
          >
            <span
              className="font-display text-[7px] tracking-[0.1em] px-1.5 py-0.5 rounded shrink-0"
              style={{ color: typeColors[tx.type], backgroundColor: typeColors[tx.type] + '15', border: `1px solid ${typeColors[tx.type]}30` }}
            >
              {tx.type.replace('_', ' ')}
            </span>
            <span className="font-mono text-[8px] text-secondary truncate flex-1">{tx.hash}</span>
            <span className="font-mono text-[8px] text-muted-foreground shrink-0">#{tx.block}</span>
            <span className="font-display text-[7px] shrink-0" style={{ color: tx.confirmed ? '#00E676' : '#F5A623' }}>
              {tx.confirmed ? '✓' : '⏳'}
            </span>
          </motion.div>
        ))}
      </div>

      {/* Bottom stats */}
      <div className="flex justify-between pt-2 border-t border-foreground/[0.06]">
        <div>
          <div className="font-display text-[7px] tracking-[0.15em] text-muted-foreground">TOTAL TXs</div>
          <div className="font-mono text-[10px] text-secondary">{txs.length}</div>
        </div>
        <div>
          <div className="font-display text-[7px] tracking-[0.15em] text-muted-foreground">BLOCK HEIGHT</div>
          <div className="font-mono text-[10px] text-secondary">{blockHeight}</div>
        </div>
        <div>
          <div className="font-display text-[7px] tracking-[0.15em] text-muted-foreground">GAS</div>
          <div className="font-mono text-[10px] text-secondary">~0.001 MATIC</div>
        </div>
      </div>
    </div>
  );
};

export default BlockchainLedger;
