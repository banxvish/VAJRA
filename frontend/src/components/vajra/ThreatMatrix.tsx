import { useVajra } from '@/context/VajraContext';
import { motion } from 'framer-motion';

const dimensions = [
  { label: 'AUDIO SYNC', authScore: 94, threatScore: 18 },
  { label: 'FACIAL MATCH', authScore: 91, threatScore: 12 },
  { label: 'LIVENESS', authScore: 97, threatScore: 8 },
  { label: 'ENTROPY', authScore: 6, threatScore: 89, inverted: true },
];

const ThreatMatrix = () => {
  const { systemStatus, threatLevel } = useVajra();

  const isThreat = systemStatus === 'threat';
  const isVerified = systemStatus === 'verified';
  const displayLevel = isThreat ? threatLevel : isVerified ? Math.max(2, 100 - threatLevel) < 10 ? threatLevel : threatLevel : 0;

  const gaugeColor = displayLevel > 70 ? '#FF3B5C' : displayLevel > 40 ? '#F5A623' : '#00E676';
  const circumference = 2 * Math.PI * 54;
  const dashOffset = circumference - (circumference * displayLevel) / 100;

  const levelLabel = displayLevel > 70 ? 'CRITICAL' : displayLevel > 40 ? 'ELEVATED' : displayLevel > 10 ? 'LOW' : 'NONE';
  const levelColor = displayLevel > 70 ? '#FF3B5C' : displayLevel > 40 ? '#F5A623' : '#00E676';

  const panelGlow = systemStatus === 'threat' ? 'threat-glow animate-pulse-threat' : systemStatus === 'verified' ? 'verified-glow' : '';

  return (
    <div className={`vajra-panel bg-card rounded-lg border border-foreground/[0.06] p-4 ${panelGlow}`}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-display text-xs tracking-[0.3em] text-foreground">THREAT MATRIX</h2>
        </div>
        <span
          className="font-display text-[8px] tracking-[0.2em] px-2 py-0.5 rounded-full border"
          style={{ color: levelColor, borderColor: levelColor + '40' }}
        >
          {levelLabel}
        </span>
      </div>

      <div className="flex gap-4 items-center">
        {/* Circular gauge */}
        <div className="relative flex-shrink-0">
          <svg width="120" height="120" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="54" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="6" />
            <motion.circle
              cx="60" cy="60" r="54" fill="none"
              stroke={gaugeColor}
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={circumference}
              animate={{ strokeDashoffset: dashOffset }}
              transition={{ duration: 0.8 }}
              transform="rotate(-90 60 60)"
              style={{ filter: `drop-shadow(0 0 6px ${gaugeColor}40)` }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="font-mono text-xl font-bold" style={{ color: gaugeColor }}>{displayLevel}</span>
            <span className="font-display text-[7px] tracking-[0.2em] text-muted-foreground">THREAT %</span>
          </div>
        </div>

        {/* Dimensional scores */}
        <div className="flex-1 space-y-2">
          {dimensions.map((d) => {
            const val = isThreat ? d.threatScore : isVerified ? d.authScore : 0;
            const color = d.inverted
              ? (val > 50 ? '#FF3B5C' : '#00E676')
              : (val > 70 ? '#00E676' : val > 40 ? '#F5A623' : '#FF3B5C');
            return (
              <div key={d.label}>
                <div className="flex justify-between mb-0.5">
                  <span className="font-display text-[7px] tracking-[0.15em] text-muted-foreground">{d.label}</span>
                  <span className="font-mono text-[9px]" style={{ color }}>{val}%</span>
                </div>
                <div className="h-1 bg-background/50 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    animate={{ width: `${val}%` }}
                    transition={{ duration: 0.5 }}
                    style={{ backgroundColor: color }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ThreatMatrix;
