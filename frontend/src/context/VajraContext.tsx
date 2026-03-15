import React, { createContext, useContext, useState, ReactNode } from 'react';

export type SystemStatus = 'idle' | 'active' | 'threat' | 'verified';

export interface AnalysisResult {
  trust_score: number;
  status: 'SAFE' | 'SUSPICIOUS' | 'FAKE';
  models: {
    spectrogram: string;
    wav2vec: string;
    codec: string;
    speaker_similarity: number;
  };
}

interface VajraState {
  systemStatus: SystemStatus;
  threatLevel: number;
  setSystemStatus: (status: SystemStatus) => void;
  setThreatLevel: (level: number) => void;
  analysisResult: AnalysisResult | null;
  setAnalysisResult: (result: AnalysisResult | null) => void;
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
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  return (
    <VajraContext.Provider value={{ systemStatus, threatLevel, setSystemStatus, setThreatLevel, analysisResult, setAnalysisResult }}>
      {children}
    </VajraContext.Provider>
  );
};
