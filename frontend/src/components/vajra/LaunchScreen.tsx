import { motion } from 'framer-motion';
import { Zap } from 'lucide-react';

interface LaunchScreenProps {
  onActivate: () => void;
}

const badges = ['VOICE AI', 'ADVERSARIAL DEFENSE', 'ZK ATTESTATION', 'POLYGON BLOCKCHAIN'];
const indicators = [
  { label: 'VOICE AI', color: '#00E676' },
  { label: 'ADVERSARIAL', color: '#00E676' },
  { label: 'ZK PROOF', color: '#00E676' },
  { label: 'BLOCKCHAIN', color: '#00E676' },
];

const LaunchScreen = ({ onActivate }: LaunchScreenProps) => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center relative z-10 px-4">
      {/* Lightning with rotating ring */}
      <motion.div
        className="relative mb-8"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: 'spring', duration: 0.6 }}
      >
        {/* Rotating ring */}
        <div className="w-48 h-48 flex items-center justify-center relative">
          <svg className="absolute inset-0 w-full h-full animate-rotate-ring" viewBox="0 0 200 200">
            <defs>
              <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#F5A623" stopOpacity="1" />
                <stop offset="50%" stopColor="#F5A623" stopOpacity="0.1" />
                <stop offset="100%" stopColor="#F5A623" stopOpacity="0.8" />
              </linearGradient>
            </defs>
            <circle
              cx="100" cy="100" r="90"
              fill="none"
              stroke="url(#ringGrad)"
              strokeWidth="2"
              strokeDasharray="150 420"
              strokeLinecap="round"
            />
          </svg>
          <Zap className="w-20 h-20 text-primary fill-primary/20" />
        </div>
      </motion.div>

      {/* Title */}
      <motion.h1
        className="font-display text-6xl md:text-8xl font-black tracking-[0.3em] text-primary mb-2"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        VAJRA
      </motion.h1>

      <motion.p
        className="text-3xl md:text-4xl mb-4 text-foreground/60"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        वज्र
      </motion.p>

      <motion.p
        className="font-display text-xs md:text-sm tracking-[0.4em] text-muted-foreground mb-10 text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        ZERO-TRUST CRYPTOGRAPHIC IDENTITY DEFENSE
      </motion.p>

      {/* Badges */}
      <motion.div
        className="flex flex-wrap justify-center gap-3 mb-12"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        {badges.map((badge) => (
          <span
            key={badge}
            className="font-display text-[10px] tracking-[0.2em] px-4 py-1.5 rounded-full border border-primary/30 text-primary/80 bg-primary/5"
          >
            {badge}
          </span>
        ))}
      </motion.div>

      {/* CTA Button */}
      <motion.button
        onClick={onActivate}
        className="font-display text-sm tracking-[0.3em] px-10 py-4 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors flex items-center gap-3 mb-16"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.6 }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.98 }}
      >
        <Zap className="w-5 h-5" />
        ACTIVATE VAJRA
      </motion.button>

      {/* System indicators */}
      <motion.div
        className="flex gap-8 md:gap-12"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
      >
        {indicators.map((ind) => (
          <div key={ind.label} className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-full animate-pulse-dot"
              style={{ backgroundColor: ind.color }}
            />
            <span className="font-display text-[10px] tracking-[0.2em] text-muted-foreground">
              {ind.label}
            </span>
          </div>
        ))}
      </motion.div>
    </div>
  );
};

export default LaunchScreen;
