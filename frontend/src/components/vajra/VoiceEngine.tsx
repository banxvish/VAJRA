import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, UploadCloud, Square, Zap } from 'lucide-react';
import { useVajra, AnalysisResult } from '@/context/VajraContext';

const VoiceEngine = () => {
  const { systemStatus, setSystemStatus, setThreatLevel, setAnalysisResult, analysisResult: result } = useVajra();
  const [scanning, setScanning] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (file: File) => {
    if (!file) return;

    setErrorMessage("");
    setScanning(true);
    setAnalysisResult(null);
    setSystemStatus('active');
    setThreatLevel(0);

    const formData = new FormData();
    formData.append("audio", file);

    try {
      const response = await fetch("http://localhost:8000/analyze_audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      setAnalysisResult(data);

      if (data.status === 'SAFE') {
        setSystemStatus('verified');
        setThreatLevel(Math.max(0, Math.floor((1 - data.trust_score) * 100)));
      } else if (data.status === 'SUSPICIOUS') {
        setSystemStatus('threat');
        setThreatLevel(Math.floor((1 - data.trust_score) * 100));
      } else {
        setSystemStatus('threat');
        setThreatLevel(Math.min(100, Math.floor((1 - data.trust_score) * 100)));
      }
    } catch (err) {
      console.error(err);
      setErrorMessage("Analysis failed. Ensure FastAPI is running on port 8000.");
      setSystemStatus('idle');
    } finally {
      setScanning(false);
    }
  };

  const getStatusColor = (status: string) => {
    if (status === 'SAFE') return '#00E676'; // Green
    if (status === 'SUSPICIOUS') return '#F5A623'; // Yellow
    if (status === 'FAKE') return '#FF3B5C'; // Red
    return '#7A8BB5'; // Default gray/blue
  };

  const panelGlow =
    systemStatus === 'threat' ? 'threat-glow animate-pulse-threat' : systemStatus === 'verified' ? 'verified-glow' : '';

  return (
    <div className={`vajra-panel bg-card rounded-lg border border-foreground/[0.06] p-4 flex-1 flex flex-col ${panelGlow}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-display text-xs tracking-[0.3em] text-foreground">VOICE ENGINE INTERFACE</h2>
          <p className="font-display text-[8px] tracking-[0.2em] text-muted-foreground mt-0.5">
            4-MODEL ENSEMBLE · REAL-TIME
          </p>
        </div>
        <Mic className="w-5 h-5 text-primary" />
      </div>

      {/* Analysis UI */}
      <AnimatePresence>
        {scanning && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mb-4"
          >
            <div className="flex flex-col items-center justify-center gap-2 py-4 rounded-md bg-primary/10 border border-primary/20">
              <Zap className="w-6 h-6 text-primary animate-pulse" />
              <span className="font-display text-[10px] tracking-[0.2em] text-primary">⟳ EXTRACTING FEATURES & RUNNING MODELS...</span>
            </div>
          </motion.div>
        )}

        {errorMessage && (
          <div className="mb-4 flex items-center justify-center gap-2 py-2 rounded-md bg-destructive/10 border border-destructive/20">
            <span className="font-display text-[10px] tracking-[0.2em] text-destructive">{errorMessage}</span>
          </div>
        )}

        {result && !scanning && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-4"
          >
            <div
              className="p-4 rounded-md border"
              style={{
                borderColor: getStatusColor(result.status),
                backgroundColor: `${getStatusColor(result.status)}1A`, // 1A is ~10% opacity
              }}
            >
              <div className="flex justify-between items-center mb-3">
                <span className="font-display text-xs tracking-[0.2em] font-bold" style={{ color: getStatusColor(result.status) }}>
                  OVERALL STATUS: {result.status}
                </span>
                <span className="font-mono text-xl font-bold" style={{ color: getStatusColor(result.status) }}>
                  {result.trust_score.toFixed(2)}
                </span>
              </div>
              <div className="w-full bg-background/50 h-2 rounded-full overflow-hidden mb-4">
                <motion.div
                  className="h-full rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${result.trust_score * 100}%` }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                  style={{ backgroundColor: getStatusColor(result.status) }}
                />
              </div>

              {/* Individual Model Results */}
              <div className="grid grid-cols-2 gap-2 text-[10px] font-display tracking-widest text-muted-foreground">
                <div className="flex justify-between items-center bg-background/30 p-2 rounded">
                  <span>SPECTROGRAM</span>
                  <span style={{ color: result.models.spectrogram === 'REAL' ? '#00E676' : '#FF3B5C', fontWeight: 'bold' }}>
                    {result.models.spectrogram}
                  </span>
                </div>
                <div className="flex justify-between items-center bg-background/30 p-2 rounded">
                  <span>WAV2VEC2</span>
                  <span style={{ color: result.models.wav2vec === 'REAL' ? '#00E676' : '#FF3B5C', fontWeight: 'bold' }}>
                    {result.models.wav2vec}
                  </span>
                </div>
                <div className="flex justify-between items-center bg-background/30 p-2 rounded">
                  <span>CODEC</span>
                  <span style={{ color: result.models.codec === 'HUMAN' ? '#00E676' : '#FF3B5C', fontWeight: 'bold' }}>
                    {result.models.codec}
                  </span>
                </div>
                <div className="flex justify-between items-center bg-background/30 p-2 rounded">
                  <span>SPEAKER MATCH</span>
                  <span style={{ 
                    color: result.models.speaker_similarity >= 0.70 ? '#00E676' : result.models.speaker_similarity > 0.40 ? '#F5A623' : '#FF3B5C', 
                    fontWeight: 'bold' 
                  }}>
                    {result.models.speaker_similarity >= 0.70 ? 'HIGH' : result.models.speaker_similarity > 0.40 ? 'MEDIUM' : 'LOW'} ({result.models.speaker_similarity.toFixed(2)})
                  </span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-auto pt-4 flex gap-2">
        <input 
          type="file" 
          ref={fileInputRef} 
          accept="audio/*" 
          className="hidden" 
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFileUpload(e.target.files[0]);
            }
          }} 
        />
        {!scanning ? (
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 font-display text-[10px] tracking-[0.2em] py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors flex items-center justify-center gap-2"
          >
            <UploadCloud className="w-4 h-4" /> UPLOAD AUDIO FILE
          </button>
        ) : (
          <button
            disabled
            className="flex-1 font-display text-[10px] tracking-[0.2em] py-3 bg-muted text-muted-foreground rounded-md flex items-center justify-center gap-2 cursor-not-allowed"
          >
            <Square className="w-4 h-4" /> PROCESSING...
          </button>
        )}
      </div>
    </div>
  );
};

export default VoiceEngine;
