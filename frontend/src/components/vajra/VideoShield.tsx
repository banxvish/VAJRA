import { useState } from 'react';
import { Camera, Shield, Square } from 'lucide-react';
import { useVajra } from '@/context/VajraContext';

const VideoShield = () => {
  const { systemStatus } = useVajra();
  const [cameraOn, setCameraOn] = useState(false);
  const [shieldActive, setShieldActive] = useState(false);

  const panelGlow = systemStatus === 'threat' ? 'threat-glow animate-pulse-threat' : systemStatus === 'verified' ? 'verified-glow' : '';

  return (
    <div className={`vajra-panel bg-card rounded-lg border border-foreground/[0.06] p-4 ${panelGlow}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="font-display text-xs tracking-[0.3em] text-foreground">VIDEO SHIELD</h2>
          <div className="flex gap-2 mt-1">
            {['FGSM', 'PGD', 'ADVERSARIAL'].map(b => (
              <span key={b} className="font-display text-[7px] tracking-[0.15em] px-2 py-0.5 rounded-full border border-secondary/30 text-secondary/70">
                {b}
              </span>
            ))}
          </div>
        </div>
        <Shield className="w-5 h-5 text-secondary" />
      </div>

      {/* Video area */}
      <div className="relative aspect-video bg-background/80 rounded-md mb-3 overflow-hidden">
        {/* HUD corners */}
        <div className="absolute top-2 left-2 w-6 h-6 border-t-2 border-l-2 border-secondary/60" />
        <div className="absolute top-2 right-2 w-6 h-6 border-t-2 border-r-2 border-secondary/60" />
        <div className="absolute bottom-2 left-2 w-6 h-6 border-b-2 border-l-2 border-secondary/60" />
        <div className="absolute bottom-2 right-2 w-6 h-6 border-b-2 border-r-2 border-secondary/60" />

        {!cameraOn ? (
          <div className="flex items-center justify-center h-full">
            <span className="font-display text-[10px] tracking-[0.2em] text-muted-foreground">NO FEED</span>
          </div>
        ) : (
          <>
            {shieldActive && (
              <>
                <div className="absolute inset-0 noise-overlay opacity-30" />
                <div className="absolute inset-0 scanline-overlay" />
              </>
            )}
            <div className="flex items-center justify-center h-full">
              <div className="w-12 h-12 rounded-full bg-secondary/10 border border-secondary/20 flex items-center justify-center">
                <Camera className="w-6 h-6 text-secondary/50" />
              </div>
            </div>
          </>
        )}

        {/* Shield badge */}
        {shieldActive && (
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-background/80 px-3 py-1 rounded-full border border-secondary/20">
            <span className="font-display text-[8px] tracking-[0.15em] text-secondary">
              🛡️ SHIELD ACTIVE · ε=12
            </span>
          </div>
        )}
      </div>

      {/* Stats */}
      {shieldActive && (
        <div className="flex justify-center gap-4 mb-3 text-center">
          {[
            { l: 'ε STRENGTH', v: '12px' },
            { l: 'ALGORITHM', v: 'FGSM+PGD' },
            { l: 'DEEPFAKES', v: 'COLLAPSED' },
          ].map(s => (
            <div key={s.l}>
              <div className="font-display text-[7px] tracking-[0.15em] text-muted-foreground">{s.l}</div>
              <div className="font-mono text-[10px] text-secondary">{s.v}</div>
            </div>
          ))}
        </div>
      )}

      {/* Button */}
      {!cameraOn ? (
        <button
          onClick={() => setCameraOn(true)}
          className="w-full font-display text-[10px] tracking-[0.2em] py-2.5 bg-secondary/10 text-secondary border border-secondary/20 rounded-md hover:bg-secondary/20 transition-colors flex items-center justify-center gap-2"
        >
          <Camera className="w-3.5 h-3.5" /> ENABLE CAMERA
        </button>
      ) : !shieldActive ? (
        <button
          onClick={() => setShieldActive(true)}
          className="w-full font-display text-[10px] tracking-[0.2em] py-2.5 bg-secondary/10 text-secondary border border-secondary/20 rounded-md hover:bg-secondary/20 transition-colors flex items-center justify-center gap-2"
        >
          <Shield className="w-3.5 h-3.5" /> ACTIVATE ADVERSARIAL SHIELD
        </button>
      ) : (
        <button
          onClick={() => { setShieldActive(false); setCameraOn(false); }}
          className="w-full font-display text-[10px] tracking-[0.2em] py-2.5 bg-muted text-muted-foreground rounded-md hover:bg-muted/80 transition-colors flex items-center justify-center gap-1"
        >
          <Square className="w-3 h-3" /> DEACTIVATE SHIELD
        </button>
      )}
    </div>
  );
};

export default VideoShield;
