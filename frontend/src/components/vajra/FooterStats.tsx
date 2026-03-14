const stats = [
  { label: 'VOICE LATENCY', value: '42ms', color: '#00E676' },
  { label: 'MODEL ENSEMBLE', value: '3-LAYER', color: '#F5A623' },
  { label: 'ZK CIRCUIT', value: 'RISC ZERO', color: '#7B61FF' },
  { label: 'BLOCKCHAIN', value: 'POLYGON AMOY', color: '#00E5FF' },
  { label: 'SESSIONS', value: '1', color: '#E8ECFF' },
  { label: 'UPTIME', value: '99.9%', color: '#00E676' },
];

const FooterStats = () => {
  return (
    <div className="border-t border-foreground/[0.06] px-4 py-3">
      <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
        {stats.map((s) => (
          <div key={s.label} className="text-center">
            <div className="font-display text-[7px] tracking-[0.2em] text-muted-foreground">{s.label}</div>
            <div className="font-mono text-[11px] font-semibold" style={{ color: s.color }}>{s.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FooterStats;
