import React, { useState } from 'react';

interface FocalRadii {
  global: number;
  self: number;
  memory: number;
}

interface Props {
  layers: number[];
  setLayers: (newLayers: number[]) => void;
  focalRadii: FocalRadii;
  setFocalRadii: React.Dispatch<React.SetStateAction<FocalRadii>>;
}

export const ModelConfig: React.FC<Props> = ({ layers, setLayers, focalRadii, setFocalRadii }) => {
  const [targetDepth, setTargetDepth] = useState(3);

  // Hexes in radius R = 3R(R+1) + 1
  const globalHexes = 3 * focalRadii.global * (focalRadii.global + 1) + 1;
  const selfHexes = 3 * focalRadii.self * (focalRadii.self + 1) + 1;
  const memoryHexes = 3 * focalRadii.memory * (focalRadii.memory + 1) + 1;
  
  const HEX_INPUTS = globalHexes + selfHexes + (memoryHexes * 4);
  const CONTEXT_INPUTS = 4; // Team, 1, 0, Turn
  const LOCALIZATION_INPUTS = 12; // [Q, R] for all 6 focal windows
  const INPUT_NODES = HEX_INPUTS + CONTEXT_INPUTS + LOCALIZATION_INPUTS; 

  // Output: Mirror selection of hexes for 2 moves
  const OUTPUT_NODES = HEX_INPUTS * 2; 

  const updateLayer = (index: number, val: number) => {
    const newLayers = [...layers];
    newLayers[index] = Math.max(1, Math.min(2048, val ?? 1));
    setLayers(newLayers);
  };

  const addLayer = () => {
    if (layers.length < 15) {
      setLayers([...layers, 256]);
    }
  };

  const removeLayer = (index: number) => {
    if (layers.length > 1) {
      const newLayers = layers.filter((_, i) => i !== index);
      setLayers(newLayers);
    }
  };

  const applyRecommended = () => {
    const newLayers: number[] = [];
    let currentSize = INPUT_NODES > 2000 ? 1024 : 512;
    for (let i = 0; i < targetDepth; i++) {
      newLayers.push(currentSize);
      if (i === targetDepth - 2) {
        currentSize = 512;
      } else {
        currentSize = Math.max(256, Math.floor(currentSize * 0.7));
      }
    }
    setLayers(newLayers);
  };

  const updateRadius = (key: keyof FocalRadii, val: number) => {
    setFocalRadii(prev => ({ ...prev, [key]: val }));
  };

  return (
    <div className="tab-content model-config-view">
      <div className="settings-header">
        <h2>Model Architecture</h2>
        <p className="section-desc">Design the deep learning core. Configure vision depth and processing power.</p>
      </div>
      
      <div className="config-layout">
        <section className="fixed-layers card">
          <div className="layer-type-group">
            <h3>Input: Vision + Location</h3>
            <div className="fixed-node-badge">{INPUT_NODES.toLocaleString()} Nodes</div>
            <p><strong>{HEX_INPUTS.toLocaleString()}</strong> spatial nodes</p>
            <p><strong>{LOCALIZATION_INPUTS + CONTEXT_INPUTS}</strong> metadata nodes (Local + Team, 1, 0, Turn)</p>
            
            <div className="mini-input" style={{ marginTop: '20px' }}>
              <label>Global Focus (Radius {focalRadii.global})</label>
              <input type="number" value={focalRadii.global} onChange={e => updateRadius('global', parseInt(e.target.value))} min="5" max="50" />
            </div>

            <div className="mini-input" style={{ marginTop: '15px' }}>
              <label>Self Focus (Radius {focalRadii.self})</label>
              <input type="number" value={focalRadii.self} onChange={e => updateRadius('self', parseInt(e.target.value))} min="1" max="15" />
            </div>

            <div className="mini-input" style={{ marginTop: '15px' }}>
              <label>Tactical Memory (Radius {focalRadii.memory})</label>
              <input type="number" value={focalRadii.memory} onChange={e => updateRadius('memory', parseInt(e.target.value))} min="1" max="15" />
            </div>
          </div>
          
          <div className="layer-type-group" style={{ marginTop: '30px' }}>
            <h3>Output: Mirror Selection</h3>
            <div className="fixed-node-badge">{OUTPUT_NODES.toLocaleString()} Nodes</div>
            <p>Direct mapping to all visible pools for 2 moves.</p>
          </div>
        </section>

        <section className="layer-controls card">
          <div className="section-header-row">
            <h3>Hidden Layers</h3>
            <div className="recommend-controls">
              <div className="depth-input">
                <label>Depth:</label>
                <input 
                  type="number" 
                  value={targetDepth} 
                  onChange={e => setTargetDepth(Math.max(1, Math.min(15, parseInt(e.target.value))))}
                />
              </div>
              <button className="recommend-btn" onClick={applyRecommended}>Auto-Fill</button>
            </div>
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
            <button className="add-layer-btn" onClick={addLayer} disabled={layers.length >= 15}>
              + Add Layer Manually
            </button>
          </div>
        </section>
      </div>
    </div>
  );
};
