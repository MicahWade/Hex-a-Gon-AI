import React, { useState, useRef, useEffect } from 'react';

export const AITraining: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [generations] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

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

  // Neural Network Visualizer Logic (Placeholder rendering)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawModel = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const layers = [4, 8, 8, 4]; // Visual representation of layers
      const nodeRadius = 8;
      const layerSpacing = canvas.width / (layers.length + 1);
      
      // Draw Connections
      ctx.strokeStyle = 'rgba(52, 152, 219, 0.2)';
      ctx.lineWidth = 1;
      for (let i = 0; i < layers.length - 1; i++) {
        const x1 = layerSpacing * (i + 1);
        const x2 = layerSpacing * (i + 2);
        for (let n1 = 0; n1 < layers[i]; n1++) {
          for (let n2 = 0; n2 < layers[i+1]; n2++) {
            const y1 = (canvas.height / (layers[i] + 1)) * (n1 + 1);
            const y2 = (canvas.height / (layers[i+1] + 1)) * (n2 + 1);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      }

      // Draw Nodes
      layers.forEach((nodes, i) => {
        const x = layerSpacing * (i + 1);
        for (let j = 0; j < nodes; j++) {
          const y = (canvas.height / (nodes + 1)) * (j + 1);
          ctx.beginPath();
          ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
          ctx.fillStyle = isTraining ? '#2ecc71' : '#3498db';
          ctx.shadowBlur = 10;
          ctx.shadowColor = ctx.fillStyle;
          ctx.fill();
          ctx.shadowBlur = 0;
          ctx.strokeStyle = '#ecf0f1';
          ctx.stroke();
        }
      });
    };

    drawModel();
  }, [isTraining]);

  return (
    <div className="tab-content ai-view">
      <h2>AI Training Lab</h2>
      
      <div className="ai-grid">
        <div className="ai-main-col">
          <section className="model-viz card">
            <h3>Neural Network Architecture</h3>
            <p className="section-desc">Real-time activation heatmap of the CNN policy head.</p>
            <canvas ref={canvasRef} width={600} height={300} className="viz-canvas" />
          </section>

          <section className="training-log card">
            <h3>Training Log</h3>
            <div className="log-window">
              <p>[System] TensorFlow.js initialized...</p>
              <p>[System] GPU Acceleration: Enabled</p>
              <p>[System] Ready for training.</p>
            </div>
          </section>
        </div>

        <div className="ai-side-col">
          <section className="reward-config card">
            <h3>Reward System</h3>
            <p className="section-desc">Adjust training incentives.</p>
            
            <div className="reward-inputs">
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
            </div>
          </section>

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
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
