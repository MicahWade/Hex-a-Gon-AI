import React, { useRef, useEffect } from 'react';

interface Props {
  isTraining: boolean;
}

export const NetworkViz: React.FC<Props> = ({ isTraining }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawModel = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // More detailed architecture for the dedicated tab
      const layers = [16, 32, 32, 16, 8]; 
      const nodeRadius = 6;
      const layerSpacing = canvas.width / (layers.length + 1);
      
      // Draw Connections
      ctx.strokeStyle = isTraining ? 'rgba(46, 204, 113, 0.15)' : 'rgba(52, 152, 219, 0.1)';
      ctx.lineWidth = 0.5;
      for (let i = 0; i < layers.length - 1; i++) {
        const x1 = layerSpacing * (i + 1);
        const x2 = layerSpacing * (i + 2);
        for (let n1 = 0; n1 < layers[i]; n1++) {
          // Optimization: only draw some connections for very large layers to keep it clean
          const step = layers[i] > 20 ? 2 : 1;
          if (n1 % step !== 0) continue;

          for (let n2 = 0; n2 < layers[i+1]; n2++) {
            if (n2 % step !== 0) continue;
            
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
          
          // Activation simulation
          const isActive = isTraining && Math.random() > 0.7;
          ctx.fillStyle = isActive ? '#2ecc71' : '#3498db';
          
          if (isActive) {
            ctx.shadowBlur = 15;
            ctx.shadowColor = '#2ecc71';
          }
          
          ctx.fill();
          ctx.shadowBlur = 0;
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
          ctx.stroke();
        }
      });
    };

    let animationFrame: number;
    const animate = () => {
      drawModel();
      if (isTraining) {
        animationFrame = requestAnimationFrame(animate);
      }
    };

    if (isTraining) {
      animate();
    } else {
      drawModel();
    }

    return () => cancelAnimationFrame(animationFrame);
  }, [isTraining]);

  return (
    <div className="tab-content network-viz-view">
      <div className="settings-header">
        <h2>Neural Network Architecture</h2>
        <p className="section-desc">Deep Convolutional Policy Head Visualization</p>
      </div>
      
      <div className="card viz-container">
        <canvas ref={canvasRef} width={800} height={500} className="full-viz-canvas" />
      </div>

      <div className="network-info grid-3">
        <div className="card info-stat">
          <h4>Input Layer</h4>
          <p>13x13x3 Tensor</p>
          <span>Current Board State Crop</span>
        </div>
        <div className="card info-stat">
          <h4>Hidden Layers</h4>
          <p>3 Convolutional + 2 Dense</p>
          <span>Pattern Detection & Strategy</span>
        </div>
        <div className="card info-stat">
          <h4>Output Layer</h4>
          <p>169 Softmax Policy</p>
          <span>Move Probability Map</span>
        </div>
      </div>
    </div>
  );
};
