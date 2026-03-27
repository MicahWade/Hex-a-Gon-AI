import React, { useState } from 'react';

interface Props {
  layers: number[];
  setLayers: (newLayers: number[]) => void;
}

export const ModelConfig: React.FC<Props> = ({ layers, setLayers }) => {
  const [focalRadius, setFocalRadius] = useState(30);
  const [selfRadius, setSelfRadius] = useState(7);
  const [memoryRadius, setMemoryRadius] = useState(5);

  // Hexes in radius R = 3R(R+1) + 1
  const globalHexes = 3 * focalRadius * (focalRadius + 1) + 1;
  const selfHexes = 3 * selfRadius * (selfRadius + 1) + 1;
  const memoryHexes = 3 * memoryRadius * (memoryRadius + 1) + 1;
  
  // Input breakdown: 
  // Global (1) + Self (1) + P1 History (2) + P2 History (2) = 6 Windows
  const HEX_INPUTS = globalHexes + selfHexes + (memoryHexes * 4);
  const CONTEXT_INPUTS = 3;
  const INPUT_NODES = HEX_INPUTS + CONTEXT_INPUTS; 

  // Output: Mirror selection of all visible hexes for 2 moves
  const OUTPUT_NODES = HEX_INPUTS * 2; 

  const updateLayer = (index: number, val: number) => {
    const newLayers = [...layers];
    newLayers[index] = Math.max(1, Math.min(2048, val ?? 1));
    setLayers(newLayers);
  };

  const addLayer = () => {
    if (layers.length < 12) {
      const lastSize = layers[layers.length - 1] || 512;
      setLayers([...layers, Math.max(128, Math.floor(lastSize / 2))]);
    }
  };

  const removeLayer = (index: number) => {
    if (layers.length > 1) {
      const newLayers = layers.filter((_, i) => i !== index);
      setLayers(newLayers);
    }
  };

  const applyRecommended = () => {
    if (INPUT_NODES > 3000) {
      setLayers([1024, 1024, 512, 256]);
    } else if (INPUT_NODES > 1500) {
      setLayers([512, 512, 256]);
    } else {
      setLayers([256, 128]);
    }
  };

  return (
    <div className="tab-content model-config-view">
      <div className="settings-header">
        <h2>Model Architecture</h2>
        <p className="section-desc">Multi-Focal Vision: Tracking global battle, self focus, and player history.</p>
      </div>
      
      <div className="config-layout">
        <section className="fixed-layers card">
          <div className="layer-type-group">
            <h3>Input: Multi-Focal Eyes</h3>
            <div className="fixed-node-badge">{INPUT_NODES.toLocaleString()} Nodes</div>
            
            <div className="mini-input">
              <label>Global Focus (Radius {focalRadius})</label>
              <input type="number" value={focalRadius} onChange={e => setFocalRadius(parseInt(e.target.value))} min="5" max="50" />
              <span className="node-unit">{globalHexes} hexes</span>
            </div>

            <div className="mini-input" style={{ marginTop: '15px' }}>
              <label>Self Focus (Radius {selfRadius})</label>
              <input type="number" value={selfRadius} onChange={e => setSelfRadius(parseInt(e.target.value))} min="1" max="15" />
              <span className="node-unit">{selfHexes} hexes</span>
            </div>

            <div className="mini-input" style={{ marginTop: '15px' }}>
              <label>Tactical Memory (Radius {memoryRadius})</label>
              <input type="number" value={memoryRadius} onChange={e => setMemoryRadius(parseInt(e.target.value))} min="1" max="15" />
              <p className="node-unit" style={{ color: 'var(--text-secondary)', marginTop: '5px' }}>
                Tracks last 2 moves for P1 and P2.<br/>
                Total: {memoryHexes * 4} hexes
              </p>
            </div>
          </div>
          
          <div className="layer-type-group" style={{ marginTop: '30px' }}>
            <h3>Output: Combined Selection</h3>
            <div className="fixed-node-badge">{OUTPUT_NODES.toLocaleString()} Nodes</div>
            <p>AI chooses Move 1 & 2 from the combined pool of all visible focal windows.</p>
          </div>
        </section>

        <section className="layer-controls card">
          <div className="section-header-row">
            <h3>Hidden Layers</h3>
            <button className="recommend-btn" onClick={applyRecommended}>Recommended</button>
          </div>
          <p className="section-desc">Complexity: <strong>~{(INPUT_NODES * layers[0] + layers.reduce((acc, val, i) => acc + (layers[i+1] ? val * layers[i+1] : 0), 0) + layers[layers.length-1] * OUTPUT_NODES).toLocaleString()}</strong> Parameters</p>
          
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
        </section>
      </div>
    </div>
  );
};
