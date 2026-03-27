import React, { useState } from 'react';

interface Props {
  layers: number[];
  setLayers: (newLayers: number[]) => void;
}

export const ModelConfig: React.FC<Props> = ({ layers, setLayers }) => {
  const [focalRadius, setFocalRadius] = useState(30);

  // Hexes in radius R = 3R(R+1) + 1
  const hexCount = 3 * focalRadius * (focalRadius + 1) + 1;
  
  // Input: One-hot or state-based grid of the focal window
  const INPUT_NODES = hexCount; 
  // Output: 2 moves, each a RingIndex [Ring, Index] selection within the window
  // We'll estimate the output based on the focal radius range
  const RING_NODES = focalRadius + 1;
  const INDEX_NODES = focalRadius * 6;
  const OUTPUT_NODES = (RING_NODES + INDEX_NODES) * 2; 

  const updateLayer = (index: number, val: number) => {
    const newLayers = [...layers];
    newLayers[index] = Math.max(1, Math.min(1024, val ?? 1));
    setLayers(newLayers);
  };

  const addLayer = () => {
    if (layers.length < 12) {
      setLayers([...layers, 256]);
    }
  };

  const removeLayer = (index: number) => {
    if (layers.length > 1) {
      const newLayers = layers.filter((_, i) => i !== index);
      setLayers(newLayers);
    }
  };

  return (
    <div className="tab-content model-config-view">
      <div className="settings-header">
        <h2>Model Architecture</h2>
        <p className="section-desc">Focal Window Spatial Encoding (Radius {focalRadius} Around Move).</p>
      </div>
      
      <div className="config-layout">
        <section className="fixed-layers card">
          <div className="layer-type-group">
            <h3>Input: Focal Window</h3>
            <div className="fixed-node-badge">{INPUT_NODES.toLocaleString()} Nodes</div>
            <p>Spatial state of all hexes within radius {focalRadius} of the latest move focus.</p>
            <div className="mini-input">
              <label>Focal Radius</label>
              <input type="number" value={focalRadius} onChange={e => setFocalRadius(parseInt(e.target.value))} min="5" max="50" />
            </div>
          </div>
          
          <div className="layer-type-group" style={{ marginTop: '30px' }}>
            <h3>Output: Ring-Index Moves</h3>
            <div className="fixed-node-badge">{OUTPUT_NODES.toLocaleString()} Nodes</div>
            <p>Predicts [Ring] and [Index] for Move 1 & Move 2 (Relative to Focus).</p>
          </div>
        </section>

        <section className="layer-controls card">
          <h3>Hidden Layers (Adjustable)</h3>
          <p className="section-desc">Processing depth. Recommended: 3-5 layers for spatial pattern detection.</p>
          
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
                  <span className="node-unit">neurons</span>
                </div>
                <button className="remove-layer-btn" onClick={() => removeLayer(i)}>&times;</button>
              </div>
            ))}
            <button className="add-layer-btn" onClick={addLayer} disabled={layers.length >= 12}>
              + Add Hidden Layer
            </button>
          </div>
          
          <div className="complexity-stats">
            <div className="stat-row">
              <span>Focal Area Size:</span>
              <strong>{hexCount.toLocaleString()} Hexes</strong>
            </div>
            <div className="stat-row">
              <span>Total Parameters:</span>
              <strong>~{(INPUT_NODES * layers[0] + layers.reduce((acc, val, i) => acc + (layers[i+1] ? val * layers[i+1] : 0), 0) + layers[layers.length-1] * OUTPUT_NODES).toLocaleString()}</strong>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
