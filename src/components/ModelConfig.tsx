import React, { useState } from 'react';

interface Props {
  layers: number[];
  setLayers: (newLayers: number[]) => void;
}

export const ModelConfig: React.FC<Props> = ({ layers, setLayers }) => {
  const [focalRadius, setFocalRadius] = useState(30);
  const [selfRadius, setSelfRadius] = useState(7);

  // Focal Window A (Global)
  const globalHexes = 3 * focalRadius * (focalRadius + 1) + 1;
  // Focal Window B (Self)
  const selfHexes = 3 * selfRadius * (selfRadius + 1) + 1;
  
  // Input: State of observed hexes
  const INPUT_NODES = globalHexes + selfHexes; 

  // Output: Mirror selection of the observed hexes for 2 moves
  const OUTPUT_NODES = INPUT_NODES * 2; 

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
    // For large mirrored IO (~3k input, ~6k output), 
    // we need high-capacity middle layers.
    if (INPUT_NODES > 2500) {
      setLayers([1024, 1024, 512, 256]);
    } else if (INPUT_NODES > 1000) {
      setLayers([512, 512, 256]);
    } else {
      setLayers([256, 128]);
    }
  };

  return (
    <div className="tab-content model-config-view">
      <div className="settings-header">
        <h2>Model Architecture</h2>
        <p className="section-desc">Mirror Selection: AI chooses moves directly from its observed focal windows.</p>
      </div>
      
      <div className="config-layout">
        <section className="fixed-layers card">
          <div className="layer-type-group">
            <h3>Input: Observed Hexes</h3>
            <div className="fixed-node-badge">{INPUT_NODES.toLocaleString()} Nodes</div>
            
            <div className="mini-input">
              <label>Global Focus (Radius {focalRadius})</label>
              <input type="number" value={focalRadius} onChange={e => setFocalRadius(parseInt(e.target.value))} min="5" max="50" />
              <span className="node-unit">{globalHexes} possible moves</span>
            </div>

            <div className="mini-input" style={{ marginTop: '15px' }}>
              <label>Self Focus (Radius {selfRadius})</label>
              <input type="number" value={selfRadius} onChange={e => setSelfRadius(parseInt(e.target.value))} min="1" max="15" />
              <span className="node-unit">{selfHexes} possible moves</span>
            </div>
          </div>
          
          <div className="layer-type-group" style={{ marginTop: '30px' }}>
            <h3>Output: Move Selection</h3>
            <div className="fixed-node-badge">{OUTPUT_NODES.toLocaleString()} Nodes</div>
            <p>1-to-1 mapping of Move 1 and Move 2 to the observed input hexes.</p>
          </div>
        </section>

        <section className="layer-controls card">
          <div className="section-header-row">
            <h3>Hidden Layers</h3>
            <button className="recommend-btn" onClick={applyRecommended}>Recommended</button>
          </div>
          <p className="section-desc">Intermediate processing. Mirrored IO requires higher neuron capacity.</p>
          
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
              <span>Mirror Pool Size:</span>
              <strong>{INPUT_NODES.toLocaleString()} Hexes</strong>
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
