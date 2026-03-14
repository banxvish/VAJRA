import { Zap } from 'lucide-react';
import { useVajra } from '@/context/VajraContext';

const DashboardHeader = () => {
  const { systemStatus } = useVajra();

  const statusConfig = {
    idle: { label: 'SYSTEM STANDBY', color: '#7A8BB5', dot: '#7A8BB5' },
    active: { label: 'PROTECTION ACTIVE', color: '#00E5FF', dot: '#00E5FF' },
    threat: { label: '⚠ DEEPFAKE DETECTED', color: '#FF3B5C', dot: '#FF3B5C' },
    verified: { label: '✓ IDENTITY VERIFIED', color: '#00E676', dot: '#00E676' },
  };

  const status = statusConfig[systemStatus];

  return (
    <header className="flex items-center justify-between px-4 py-3 border-b border-foreground/5">
      {/* Left */}
      <div className="flex items-center gap-3">
        <Zap className="w-5 h-5 text-primary fill-primary/20" />
        <span className="font-display text-sm tracking-[0.3em] text-primary font-bold">VAJRA</span>
        <span className="font-display text-[9px] tracking-[0.2em] text-muted-foreground hidden md:inline">
          ZERO-TRUST · IDENTITY · DEFENSE
        </span>
      </div>

      {/* Center */}
      <div className="flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full ${systemStatus === 'threat' ? 'animate-pulse-dot' : ''}`}
          style={{ backgroundColor: status.dot }}
        />
        <span
          className="font-display text-[10px] tracking-[0.2em] font-semibold"
          style={{ color: status.color }}
        >
          {status.label}
        </span>
      </div>

      {/* Right */}
      <div className="flex items-center gap-3 text-right">
        <span className="font-display text-[9px] tracking-[0.15em] text-muted-foreground hidden lg:inline">
          NOVUS HACKATHON 2026 | MALLA REDDY UNIVERSITY
        </span>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-authentic animate-pulse-dot" />
          <span className="font-display text-[9px] tracking-[0.15em] text-muted-foreground">
            POLYGON AMOY
          </span>
        </div>
      </div>
    </header>
  );
};

export default DashboardHeader;
