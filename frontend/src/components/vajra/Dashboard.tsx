import DashboardHeader from './DashboardHeader';
import StatusBar from './StatusBar';
import VoiceEngine from './VoiceEngine';
import VideoShield from './VideoShield';
import ThreatMatrix from './ThreatMatrix';
import ZKAttestation from './ZKAttestation';
import BlockchainLedger from './BlockchainLedger';
import FooterStats from './FooterStats';

const Dashboard = () => {
  return (
    <div className="min-h-screen flex flex-col relative z-10">
      <DashboardHeader />
      <StatusBar />
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 p-4">
        {/* Left */}
        <div className="flex flex-col gap-4">
          <VoiceEngine />
        </div>
        {/* Center */}
        <div className="flex flex-col gap-4">
          <VideoShield />
          <ThreatMatrix />
        </div>
        {/* Right */}
        <div className="flex flex-col gap-4">
          <ZKAttestation />
          <BlockchainLedger />
        </div>
      </div>
      <FooterStats />
    </div>
  );
};

export default Dashboard;
