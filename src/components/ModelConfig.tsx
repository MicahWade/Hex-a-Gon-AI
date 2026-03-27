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
  
  const INPUT_NODES = globalHexes + selfHexes; 

  // Playable Radius is anything it can see (max of both views)
  const playRadius = Math.max(focalRadius, selfRadius);
  const RING_NODES = playRadius + 1;
  const INDEX_NODES = playRadius * 6;
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

  const applyRecommended = () => {
    // Recommendation logic based on input complexity (log-scaling)
    // For ~3000 nodes, we want a funnel approach
    if (INPUT_NODES > 2000) {
      setLayers([512, 256, 128]);
    } else if (INPUT_NODES > 1000) {
      setLayers([256, 128, 64]);
    } else {
      setLayers([128, 64]);
    }
  };

  return (
    <div className="tab-content model-config-view">
      <div className="settings-header">
        <h2>Model Architecture</h2>
        <p className="section-desc">Dual-Focal Spatial Encoding. AI can play anywhere within its focal visibility.</p>
      </div>
      
      <div className="config-layout">
        <section className="fixed-layers card">
          <div className="layer-type-group">
            <h3>Input Visibility</h3>
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
          </div>
          
          <div className="layer-type-group" style={{ marginTop: '30px' }}>
            <h3>Output Reach (Auto)</h3>
            <div className="fixed-node-badge">{OUTPUT_NODES.toLocaleString()} Nodes</div>
            <p>Selection range covers entire visibility (Radius {playRadius}).</p>
          </div>
        </section>

        <section className="layer-controls card">
          <div className="section-header-row">
            <h3>Hidden Layers</h3>
            <button className="recommend-btn" onClick={applyRecommended}>Recommended</button>
          </div>
          <p className="section-desc">Neurons per processing stage.</p>
          
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
              <span>Total Parameters:</span>
              <strong>~{(INPUT_NODES * layers[0] + layers.reduce((acc, val, i) => acc + (layers[i+1] ? val * layers[i+1] : 0), 0) + layers[layers.length-1] * OUTPUT_NODES).toLocaleString()}</strong>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
