import React, { useState } from 'react';

interface Props {
  isTraining: boolean;
  setIsTraining: (val: boolean) => void;
}

export const AITraining: React.FC<Props> = ({ isTraining, setIsTraining }) => {
  const [generations] = useState(0);

  // Reward States
  const [rewards, setRewards] = useState({
    p1Win: 1.0,
    p2Win: 1.1,
    p1Draw: 0.4,
    p2Draw: 0.6,
    threat: 0.2,
    efficiency: -0.01
  });

  const toggleTraining = () => {
    setIsTraining(!isTraining);
  };

  return (
    <div className="tab-content ai-view">
      <div className="settings-header">
        <h2>AI Training Lab</h2>
        <p className="section-desc">Training control center and incentive configuration.</p>
      </div>
      
      <div className="ai-grid">
        <div className="ai-main-col">
          <section className="training-log card">
            <h3>Training Log</h3>
            <div className="log-window">
              <p>[System] TensorFlow.js initialized...</p>
              <p>[System] GPU Acceleration: Enabled</p>
              <p>[System] Ready for training.</p>
            </div>
          </section>

          <section className="reward-config card">
            <h3>Reward System</h3>
            <p className="section-desc">Adjust training incentives for Player 1 and Player 2.</p>
            
            <div className="reward-grid">
              <div className="input-group">
                <label>P1 Win Weight</label>
                <input type="number" value={rewards.p1Win} step={0.1} onChange={e => setRewards({...rewards, p1Win: parseFloat(e.target.value)})} />
              </div>
              <div className="input-group">
                <label>P2 Win Weight</label>
                <input type="number" value={rewards.p2Win} step={0.1} onChange={e => setRewards({...rewards, p2Win: parseFloat(e.target.value)})} />
              </div>
              <div className="input-group">
                <label>P1 Draw (Max Moves)</label>
                <input type="number" value={rewards.p1Draw} step={0.1} onChange={e => setRewards({...rewards, p1Draw: parseFloat(e.target.value)})} />
              </div>
              <div className="input-group">
                <label>P2 Draw (Max Moves)</label>
                <input type="number" value={rewards.p2Draw} step={0.1} onChange={e => setRewards({...rewards, p2Draw: parseFloat(e.target.value)})} />
              </div>
              <div className="input-group">
                <label>Threat Detection</label>
                <input type="number" value={rewards.threat} step={0.05} onChange={e => setRewards({...rewards, threat: parseFloat(e.target.value)})} />
              </div>
              <div className="input-group">
                <label>Efficiency Penalty</label>
                <input type="number" value={rewards.efficiency} step={0.01} onChange={e => setRewards({...rewards, efficiency: parseFloat(e.target.value)})} />
              </div>
            </div>
          </section>
        </div>

        <div className="ai-side-col">
          <section className="training-stats card">
            <h3>Training Controls</h3>
            <div className="toggle-group">
              <label><input type="checkbox" /> Save Constantly</label>
            </div>
            <div className="toggle-group">
              <label><input type="checkbox" defaultChecked /> Save history</label>
            </div>
            
            <div className="actions" style={{ marginTop: '20px' }}>
              <button className={isTraining ? 'stop-btn' : 'start-btn'} onClick={toggleTraining}>
                {isTraining ? 'Stop' : 'Start Training'}
              </button>
            </div>

            <div className="stat-summary" style={{ marginTop: '20px' }}>
              <div className="stat-card"><span>Gen:</span> <span>{generations}</span></div>
              <div className="stat-card"><span>Loss:</span> <span>0.000</span></div>
              <div className="stat-card"><span>Win Rate:</span> <span>0%</span></div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
