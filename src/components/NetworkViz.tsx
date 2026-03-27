import React, { useRef, useEffect } from 'react';

interface Props {
  isTraining: boolean;
  layers: number[];
  setLayers: (newLayers: number[]) => void;
}

export const NetworkViz: React.FC<Props> = ({ isTraining, layers, setLayers }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawModel = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const nodeRadius = 5;
      const layerSpacing = canvas.width / (layers.length + 1);
      
      // Connections
      ctx.strokeStyle = isTraining ? 'rgba(46, 204, 113, 0.15)' : 'rgba(52, 152, 219, 0.1)';
      ctx.lineWidth = 0.5;
      for (let i = 0; i < layers.length - 1; i++) {
        const x1 = layerSpacing * (i + 1);
        const x2 = layerSpacing * (i + 2);
        
        // Limit connections shown if layer is too big to keep it clean
        const maxDraw = 12; 
        const step1 = Math.max(1, Math.ceil(layers[i] / maxDraw));
        const step2 = Math.max(1, Math.ceil(layers[i+1] / maxDraw));

        for (let n1 = 0; n1 < layers[i]; n1 += step1) {
          for (let n2 = 0; n2 < layers[i+1]; n2 += step2) {
            const y1 = (canvas.height / (Math.min(layers[i], maxDraw*step1) + 1)) * (n1 / step1 + 1);
            const y2 = (canvas.height / (Math.min(layers[i+1], maxDraw*step1) + 1)) * (n2 / step2 + 1);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      }

      // Nodes
      layers.forEach((nodes, i) => {
        const x = layerSpacing * (i + 1);
        const maxDraw = 12;
        const step = Math.max(1, Math.ceil(nodes / maxDraw));
        const nodesToShow = Math.min(nodes, maxDraw);

        for (let j = 0; j < nodesToShow; j++) {
          const y = (canvas.height / (nodesToShow + 1)) * (j + 1);
          ctx.beginPath();
          ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
          
          const isActive = isTraining && Math.random() > 0.8;
          ctx.fillStyle = isActive ? '#2ecc71' : '#3498db';
          
          if (isActive) {
            ctx.shadowBlur = 10;
            ctx.shadowColor = '#2ecc71';
          }
          
          ctx.fill();
          ctx.shadowBlur = 0;
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.stroke();
        }
        
        // Label node count
        ctx.fillStyle = '#bdc3c7';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${nodes}`, x, canvas.height - 10);
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
  }, [isTraining, layers]);

  const updateLayer = (index: number, val: number) => {
    const newLayers = [...layers];
    newLayers[index] = Math.max(1, Math.min(128, val));
    setLayers(newLayers);
  };

  const addLayer = () => {
    if (layers.length < 8) {
      setLayers([...layers, 16]);
    }
  };

  const removeLayer = (index: number) => {
    if (layers.length > 2) {
      const newLayers = layers.filter((_, i) => i !== index);
      setLayers(newLayers);
    }
  };

  return (
    <div className="tab-content network-viz-view">
      <div className="settings-header">
        <h2>Network Architect</h2>
        <p className="section-desc">Design your deep learning model's hidden layers.</p>
      </div>
      
      <div className="architect-layout">
        <div className="viz-main card">
          <canvas ref={canvasRef} width={800} height={400} className="full-viz-canvas" />
        </div>

        <div className="layer-controls card">
          <h3>Hidden Layer Config</h3>
          <div className="layer-list">
            {layers.map((nodes, i) => (
              <div key={i} className="layer-item">
                <div className="layer-info">
                  <span className="layer-tag">L{i}</span>
                  <input 
                    type="number" 
                    value={nodes} 
                    onChange={(e) => updateLayer(i, parseInt(e.target.value))}
                    min="1"
                    max="128"
                  />
                  <span className="node-unit">nodes</span>
                </div>
                <button 
                  className="remove-layer-btn"
                  onClick={() => removeLayer(i)}
                  title="Remove layer"
                  disabled={layers.length <= 2}
                >
                  &times;
                </button>
              </div>
            ))}
            <button className="add-layer-btn" onClick={addLayer} disabled={layers.length >= 8}>
              + Add Hidden Layer
            </button>
          </div>
          
          <div className="complexity-stats">
            <p>Total Nodes: <strong>{layers.reduce((a, b) => a + b, 0)}</strong></p>
            <p>Depth: <strong>{layers.length} Layers</strong></p>
          </div>
        </div>
      </div>
    </div>
  );
};
