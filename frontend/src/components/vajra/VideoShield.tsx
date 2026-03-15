import { useState, useRef, useEffect } from 'react';
import { Camera, Shield, Square, UploadCloud, Zap, Video } from 'lucide-react';
import { useVajra } from '@/context/VajraContext';
import { motion, AnimatePresence } from 'framer-motion';

type VideoAnalysisResult = {
  fake_probability: number;
  status: 'SAFE' | 'FAKE';
  frames_analyzed: number;
};

const VideoShield = () => {
  const { systemStatus, setSystemStatus, setThreatLevel } = useVajra();
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<VideoAnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const videoChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const [recording, setRecording] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [liveStatus, setLiveStatus] = useState<{ prob: number, status: string } | null>(null);
  const [isRppgActive, setIsRppgActive] = useState(false);

  useEffect(() => {
    if (cameraOn && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(err => console.error(err));
    }
  }, [cameraOn]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (recording && videoRef.current) {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      
      interval = setInterval(async () => {
        if (!videoRef.current) return;
        canvas.width = videoRef.current.videoWidth || 320;
        canvas.height = videoRef.current.videoHeight || 240;
        if (ctx && canvas.width > 0) {
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(async (blob) => {
            if (!blob) return;
            const formData = new FormData();
            formData.append("frame", blob, "frame.jpg");
            try {
              const res = await fetch("http://localhost:8000/analyze_frame", {
                method: "POST",
                body: formData,
              });
              const data = await res.json();
              if (res.ok) {
                setLiveStatus({ prob: data.fake_probability, status: data.status });
              }
            } catch (err) { }
          }, "image/jpeg", 0.8);
        }
      }, 1000); // 1 frame per second
    } else {
      setLiveStatus(null);
    }
    return () => clearInterval(interval);
  }, [recording]);

  const startCameraAndRecord = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;
      setCameraOn(true);
      
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
      mediaRecorderRef.current = mediaRecorder;
      videoChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) videoChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const videoBlob = new Blob(videoChunksRef.current, { type: 'video/webm' });
        const file = new File([videoBlob], 'recording.webm', { type: 'video/webm' });
        handleFileUpload(file);
        stream.getTracks().forEach(track => track.stop());
        setCameraOn(false);
        if (videoRef.current) videoRef.current.srcObject = null;
      };

      mediaRecorder.start();
      setRecording(true);
      setErrorMessage("");
      
      // KAVACHA: Trigger rPPG screen flash sequence for blood flow validation
      setIsRppgActive(true);
      setTimeout(() => {
        setIsRppgActive(false);
      }, 3500);

    } catch (err) {
      console.error("Camera access denied:", err);
      setErrorMessage("Camera access denied.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!file) return;

    setErrorMessage("");
    setScanning(true);
    setResult(null);
    setSystemStatus('active');

    const formData = new FormData();
    formData.append("video", file);

    try {
      const response = await fetch("http://localhost:8000/analyze_video", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      setResult(data);

      if (data.status === 'SAFE') {
        setSystemStatus('verified');
        setThreatLevel(Math.max(0, Math.floor(data.fake_probability * 100)));
      } else {
        setSystemStatus('threat');
        setThreatLevel(Math.floor(data.fake_probability * 100));
      }
    } catch (err) {
      console.error(err);
      setErrorMessage("Analysis failed. Ensure FastAPI is running on port 8000.");
      setSystemStatus('idle');
    } finally {
      setScanning(false);
    }
  };

  const panelGlow = systemStatus === 'threat' ? 'threat-glow animate-pulse-threat' : systemStatus === 'verified' ? 'verified-glow' : '';

  return (
    <div className={`vajra-panel bg-card rounded-lg border border-foreground/[0.06] p-4 flex flex-col ${panelGlow}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="font-display text-xs tracking-[0.3em] text-foreground">VIDEO SHIELD</h2>
          <div className="flex gap-2 mt-1">
            {['XCEPTION', 'DEEPFAKE DETECTOR'].map(b => (
              <span key={b} className="font-display text-[7px] tracking-[0.15em] px-2 py-0.5 rounded-full border border-secondary/30 text-secondary/70">
                {b}
              </span>
            ))}
          </div>
        </div>
        <Shield className="w-5 h-5 text-secondary" />
      </div>

      <AnimatePresence>
        {cameraOn && (
           <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4 rounded-md overflow-hidden bg-black flex flex-row items-center relative gap-2 border border-secondary/20 p-2"
           >
             <div className="flex-1 relative overflow-hidden rounded bg-black">
               <video 
                 ref={videoRef} 
                 playsInline 
                 muted 
                 className="w-full object-cover max-h-[300px]" 
               />

               {/* rPPG LIVENESS VALIDATION SEQUENCE */}
               <AnimatePresence>
                 {isRppgActive && (
                   <motion.div
                     initial={{ opacity: 0 }}
                     animate={{ opacity: 0.35, backgroundColor: ['#ff0000', '#00ff00', '#0000ff', '#ffffff'] }}
                     exit={{ opacity: 0 }}
                     transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
                     className="absolute inset-0 z-10 pointer-events-none mix-blend-screen"
                   >
                     <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-white font-mono text-[8px] bg-black/50 px-2 rounded opacity-80 backdrop-blur-sm">
                       rPPG SIGNAL LOCKING...
                     </div>
                   </motion.div>
                 )}
               </AnimatePresence>
               
               {/* SIMPLE WARNING SHIELD */}
               <AnimatePresence>
                 {liveStatus && liveStatus.status === 'FAKE' && (
                   <motion.div 
                     initial={{ opacity: 0 }}
                     animate={{ opacity: 1 }}
                     exit={{ opacity: 0 }}
                     className="absolute inset-0 pointer-events-none z-10 flex flex-col items-center justify-center bg-destructive/10"
                   >
                     <div className="flex flex-col items-center justify-center bg-black/60 px-4 py-2 rounded-lg border border-destructive/50 backdrop-blur-sm">
                        <Shield className="w-8 h-8 text-destructive mb-1" />
                        <span className="font-display font-bold text-[10px] tracking-[0.2em] text-destructive">
                          THREAT BLOCKED
                        </span>
                     </div>
                   </motion.div>
                 )}
               </AnimatePresence>

               {/* Live Status Overlay on Video Box */}
               {liveStatus && (
                 <div 
                   className="absolute top-2 left-2 px-2 py-1 rounded z-20 font-display text-[10px] font-bold tracking-[0.2em] border backdrop-blur-md shadow-lg"
                   style={{ 
                     backgroundColor: liveStatus.status === 'SAFE' ? 'rgba(0, 230, 118, 0.15)' : 'rgba(255, 59, 92, 0.15)',
                     borderColor: liveStatus.status === 'SAFE' ? 'rgba(0, 230, 118, 0.4)' : 'rgba(255, 59, 92, 0.4)',
                     color: liveStatus.status === 'SAFE' ? '#00E676' : '#FF3B5C'
                   }}
                 >
                   {liveStatus.status === 'SAFE' ? 'REAL' : 'FAKE'} • {(liveStatus.prob * 100).toFixed(0)}%
                 </div>
               )}

               {recording && (
                  <div className="absolute top-2 right-2 flex items-center gap-2 bg-black/50 px-2 py-1 rounded z-20">
                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-[10px] text-white font-mono">REC</span>
                  </div>
               )}
             </div>
             
             {/* Live Frame Analysis sidebar */}
             {recording && (
               <div className="w-[120px] flex flex-col items-center justify-center p-2 rounded bg-black/50 border border-secondary/10 shrink-0 h-full">
                 <span className="font-display text-[8px] tracking-[0.2em] text-muted-foreground mb-2 text-center">
                   LIVE REVIEW
                 </span>
                 {liveStatus ? (
                   <>
                     <div className="text-[10px] font-bold mb-1" style={{ color: liveStatus.status === 'SAFE' ? '#00E676' : '#FF3B5C' }}>
                       {liveStatus.status}
                     </div>
                     <div className="w-full bg-background/50 h-1.5 rounded-full overflow-hidden mb-2">
                       <motion.div
                         className="h-full rounded-full"
                         initial={{ width: 0 }}
                         animate={{ width: `${liveStatus.prob * 100}%` }}
                         transition={{ duration: 0.3 }}
                         style={{ backgroundColor: liveStatus.status === 'SAFE' ? '#00E676' : '#FF3B5C' }}
                       />
                     </div>
                     <span className="font-mono text-[10px]" style={{ color: liveStatus.status === 'SAFE' ? '#00E676' : '#FF3B5C' }}>
                       {(liveStatus.prob * 100).toFixed(1)}%
                     </span>
                   </>
                 ) : (
                   <div className="flex flex-col items-center">
                     <Zap className="w-4 h-4 text-secondary/50 animate-pulse mb-1" />
                     <span className="font-display text-[7px] text-secondary/50 text-center">SAMPLING...</span>
                   </div>
                 )}
               </div>
             )}
           </motion.div>
        )}
        {scanning && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mb-4"
          >
            <div className="flex flex-col items-center justify-center gap-2 py-4 rounded-md bg-secondary/10 border border-secondary/20">
              <Zap className="w-6 h-6 text-secondary animate-pulse" />
              <span className="font-display text-[10px] tracking-[0.2em] text-secondary">⟳ EXTRACTING FRAMES & RUNNING MODEL...</span>
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
            <div className="p-4 rounded-md border" style={{ borderColor: result.status === 'SAFE' ? '#00E676' : '#FF3B5C', backgroundColor: `${result.status === 'SAFE' ? '#00E676' : '#FF3B5C'}1A` }}>
              <div className="flex justify-between items-center mb-3">
                <span className="font-display text-xs tracking-[0.2em] font-bold" style={{ color: result.status === 'SAFE' ? '#00E676' : '#FF3B5C' }}>
                  OVERALL STATUS: {result.status}
                </span>
                <span className="font-mono text-xl font-bold" style={{ color: result.status === 'SAFE' ? '#00E676' : '#FF3B5C' }}>
                  {(result.fake_probability * 100).toFixed(1)}% FAKE
                </span>
              </div>
              <div className="w-full bg-background/50 h-2 rounded-full overflow-hidden mb-4">
                <motion.div
                  className="h-full rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${result.fake_probability * 100}%` }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                  style={{ backgroundColor: result.status === 'SAFE' ? '#00E676' : '#FF3B5C' }}
                />
              </div>
              
              <div className="flex justify-center gap-4 text-center">
                <div>
                  <div className="font-display text-[7px] tracking-[0.15em] text-muted-foreground">FRAMES ANALYZED</div>
                  <div className="font-mono text-[10px]" style={{ color: result.status === 'SAFE' ? '#00E676' : '#FF3B5C' }}>{result.frames_analyzed}</div>
                </div>
                <div>
                  <div className="font-display text-[7px] tracking-[0.15em] text-muted-foreground">ALGORITHM</div>
                  <div className="font-mono text-[10px]" style={{ color: result.status === 'SAFE' ? '#00E676' : '#FF3B5C' }}>XCEPTION SPATIAL</div>
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
          accept="video/*,image/*" 
          className="hidden" 
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFileUpload(e.target.files[0]);
            }
          }} 
        />
        {!scanning && !recording ? (
          <>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 font-display text-[10px] tracking-[0.2em] py-2.5 bg-secondary/10 text-secondary border border-secondary/20 rounded-md hover:bg-secondary/20 transition-colors flex items-center justify-center gap-2"
            >
              <UploadCloud className="w-3.5 h-3.5" /> UPLOAD
            </button>
            <button
              onClick={startCameraAndRecord}
              className="flex-1 font-display text-[10px] tracking-[0.2em] py-2.5 bg-red-500/10 text-red-500 border border-red-500/20 rounded-md hover:bg-red-500/20 transition-colors flex items-center justify-center gap-2"
            >
              <Video className="w-3.5 h-3.5" /> RECORD
            </button>
          </>
        ) : recording ? (
           <button
            onClick={stopRecording}
            className="w-full font-display text-[10px] tracking-[0.2em] py-2.5 bg-red-500 text-white rounded-md hover:bg-red-600 transition-colors flex items-center justify-center gap-2 animate-pulse"
          >
            <Square className="w-3 h-3" /> STOP & ANALYZE
          </button>
        ) : (
          <button
            disabled
            className="w-full font-display text-[10px] tracking-[0.2em] py-2.5 bg-muted text-muted-foreground rounded-md flex items-center justify-center gap-1 cursor-not-allowed"
          >
            <Square className="w-3 h-3" /> PROCESSING...
          </button>
        )}
      </div>
    </div>
  );
};

export default VideoShield;
