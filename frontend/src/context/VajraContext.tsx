import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export type SystemStatus = 'idle' | 'active' | 'threat' | 'verified';

interface VajraState {
  systemStatus: SystemStatus;
  threatLevel: number;
  setSystemStatus: (status: SystemStatus) => void;
  setThreatLevel: (level: number) => void;
}

const VajraContext = createContext<VajraState | null>(null);

export const useVajra = () => {
  const ctx = useContext(VajraContext);
  if (!ctx) throw new Error('useVajra must be used within VajraProvider');
  return ctx;
};

export const VajraProvider = ({ children }: { children: ReactNode }) => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>('idle');
  const [threatLevel, setThreatLevel] = useState(0);

  return (
    <VajraContext.Provider value={{ systemStatus, threatLevel, setSystemStatus, setThreatLevel }}>
      {children}
    </VajraContext.Provider>
  );
};
