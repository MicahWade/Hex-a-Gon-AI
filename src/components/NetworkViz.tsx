import React, { useRef, useEffect, useState } from 'react';

interface Props {
  isTraining: boolean;
  layers: number[];
  setLayers: (newLayers: number[]) => void;
}

export const NetworkViz: React.FC<Props> = ({ isTraining, layers, setLayers }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewport, setViewport] = useState({ x: 0, y: 0, scale: 0.8 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

  // Fixed Input and Output
  const INPUT_NODES = 169; // 13x13
  const OUTPUT_NODES = 169;
  
  // Full architecture for rendering: Input + Hidden + Output
  const fullArchitecture = [INPUT_NODES, ...layers, OUTPUT_NODES];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawModel = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      
      // Apply Pan and Zoom
      ctx.translate(canvas.width / 2 + viewport.x, canvas.height / 2 + viewport.y);
      ctx.scale(viewport.scale, viewport.scale);

      const nodeRadius = 8;
      const layerSpacing = 200;
      const totalWidth = (fullArchitecture.length - 1) * layerSpacing;
      const startX = -totalWidth / 2;
      
      // Connections
      ctx.strokeStyle = isTraining ? 'rgba(46, 204, 113, 0.1)' : 'rgba(52, 152, 219, 0.08)';
      ctx.lineWidth = 1;
      
      for (let i = 0; i < fullArchitecture.length - 1; i++) {
        const x1 = startX + i * layerSpacing;
        const x2 = startX + (i + 1) * layerSpacing;
        
        const maxDraw = 15; 
        const step1 = Math.max(1, Math.ceil(fullArchitecture[i] / maxDraw));
        const step2 = Math.max(1, Math.ceil(fullArchitecture[i+1] / maxDraw));

        for (let n1 = 0; n1 < fullArchitecture[i]; n1 += step1) {
          for (let n2 = 0; n2 < fullArchitecture[i+1]; n2 += step2) {
            const h1 = Math.min(fullArchitecture[i], maxDraw * step1);
            const h2 = Math.min(fullArchitecture[i+1], maxDraw * step1);
            const y1 = (400 / (h1/step1 + 1)) * (n1 / step1 + 1) - 200;
            const y2 = (400 / (h2/step2 + 1)) * (n2 / step2 + 1) - 200;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      }

      // Nodes
      fullArchitecture.forEach((nodes, i) => {
        const x = startX + i * layerSpacing;
        const maxDraw = 15;
        const step = Math.max(1, Math.ceil(nodes / maxDraw));
        const nodesToShow = Math.min(nodes, maxDraw);

        for (let j = 0; j < nodesToShow; j++) {
          const y = (400 / (nodesToShow + 1)) * (j + 1) - 200;
          ctx.beginPath();
          ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
          
          const isActive = isTraining && Math.random() > 0.85;
          ctx.fillStyle = isActive ? '#2ecc71' : '#3498db';
          
          if (isActive) {
            ctx.shadowBlur = 15;
            ctx.shadowColor = '#2ecc71';
          }
          
          ctx.fill();
          ctx.shadowBlur = 0;
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        
        // Labels
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText(`${nodes}`, x, 240);
        
        ctx.fillStyle = '#bdc3c7';
        ctx.font = '12px Inter, system-ui';
        let label = `Hidden L${i}`;
        if (i === 0) label = 'Input (13x13)';
        if (i === fullArchitecture.length - 1) label = 'Output (Policy)';
        ctx.fillText(label, x, 260);
      });

      ctx.restore();
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
  }, [isTraining, layers, viewport]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setLastMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      const dx = e.clientX - lastMousePos.x;
      const dy = e.clientY - lastMousePos.y;
      setViewport(v => ({ ...v, x: v.x + dx, y: v.y + dy }));
      setLastMousePos({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent) => {
    const scaleFactor = 1.1;
    const newScale = e.deltaY < 0 ? viewport.scale * scaleFactor : viewport.scale / scaleFactor;
    setViewport(v => ({ ...v, scale: Math.max(0.1, Math.min(3, newScale)) }));
  };

  const updateLayer = (index: number, val: number) => {
    const newLayers = [...layers];
    newLayers[index] = Math.max(1, Math.min(512, val));
    setLayers(newLayers);
  };

  const addLayer = () => {
    if (layers.length < 10) {
      setLayers([...layers, 32]);
    }
  };

  const removeLayer = (index: number) => {
    if (layers.length > 1) {
      const newLayers = layers.filter((_, i) => i !== index);
      setLayers(newLayers);
    }
  };

  return (
    <div className="tab-content network-viz-view">
      <div className="settings-header">
        <h2>Network Architect</h2>
        <p className="section-desc">Design hidden layers. Input (13x13) and Output (169) are fixed.</p>
      </div>
      
      <div className="architect-layout">
        <div className="viz-main card" style={{ overflow: 'hidden', cursor: isDragging ? 'grabbing' : 'grab' }}>
          <canvas 
            ref={canvasRef} 
            width={800} 
            height={600} 
            className="full-viz-canvas"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onWheel={handleWheel}
          />
          <div className="viz-controls-hint">Drag to Pan • Scroll to Zoom</div>
        </div>

        <div className="layer-controls card">
          <h3>Hidden Layers</h3>
          <div className="layer-list">
            {layers.map((nodes, i) => (
              <div key={i} className="layer-item">
                <div className="layer-info">
                  <span className="layer-tag">H{i+1}</span>
                  <input 
                    type="number" 
                    value={nodes} 
                    onChange={(e) => updateLayer(i, parseInt(e.target.value))}
                  />
                  <span className="node-unit">nodes</span>
                </div>
                <button className="remove-layer-btn" onClick={() => removeLayer(i)}>&times;</button>
              </div>
            ))}
            <button className="add-layer-btn" onClick={addLayer} disabled={layers.length >= 10}>
              + Add Hidden Layer
            </button>
          </div>
          
          <div className="complexity-stats">
            <p>Hidden Nodes: <strong>{layers.reduce((a, b) => a + b, 0)}</strong></p>
            <p>Total Layers: <strong>{fullArchitecture.length}</strong></p>
          </div>
        </div>
      </div>
    </div>
  );
};
