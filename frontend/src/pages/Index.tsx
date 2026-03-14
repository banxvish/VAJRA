import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import LaunchScreen from '@/components/vajra/LaunchScreen';
import Dashboard from '@/components/vajra/Dashboard';
import { VajraProvider } from '@/context/VajraContext';

const Index = () => {
  const [launched, setLaunched] = useState(false);

  return (
    <VajraProvider>
      <div className="min-h-screen relative overflow-hidden">
        {/* Top radial glow */}
        <div 
          className="fixed inset-0 pointer-events-none z-0"
          style={{
            background: 'radial-gradient(ellipse 60% 30% at 50% 0%, rgba(245,166,35,0.08), transparent)',
          }}
        />
        <AnimatePresence mode="wait">
          {!launched ? (
            <motion.div
              key="launch"
              initial={{ opacity: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3 }}
            >
              <LaunchScreen onActivate={() => setLaunched(true)} />
            </motion.div>
          ) : (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
            >
              <Dashboard />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </VajraProvider>
  );
};

export default Index;
