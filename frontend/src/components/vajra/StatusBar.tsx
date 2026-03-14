import { useEffect, useState } from 'react';
import { useVajra } from '@/context/VajraContext';

const StatusBar = () => {
  const { systemStatus } = useVajra();
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(t);
  }, []);

  const fmt = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, '0');
    const ss = (s % 60).toString().padStart(2, '0');
    return `${m}:${ss}`;
  };

  return (
    <div className="relative h-8 border-b border-foreground/5 flex items-center px-4 overflow-hidden">
      {/* Scanning line */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="h-full w-32 animate-scan-line"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(245,166,35,0.15), rgba(0,229,255,0.1), transparent)',
          }}
        />
      </div>

      <div className="flex items-center gap-2 z-10">
        <div className="w-1.5 h-1.5 rounded-full bg-authentic animate-pulse-dot" />
        <span className="font-display text-[9px] tracking-[0.2em] text-muted-foreground">
          RAKSHA ENGINE ONLINE
        </span>
      </div>

      <div className="flex-1 text-center z-10">
        <span className="font-display text-[9px] tracking-[0.3em]" style={{
          color: systemStatus === 'threat' ? '#FF3B5C' : systemStatus === 'verified' ? '#00E676' : '#F5A623'
        }}>
          {systemStatus === 'idle' ? 'VAJRA STANDBY' : 
           systemStatus === 'threat' ? 'THREAT DETECTED' :
           systemStatus === 'verified' ? 'VAJRA PROTECTION ACTIVE' : 'VAJRA SCANNING'}
        </span>
      </div>

      <div className="z-10">
        <span className="font-mono text-[10px] text-muted-foreground">{fmt(elapsed)}</span>
      </div>
    </div>
  );
};

export default StatusBar;
