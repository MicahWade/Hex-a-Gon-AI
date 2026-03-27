import React from 'react';

interface Props {
  layers: number[];
  setLayers: (newLayers: number[]) => void;
}

export const ModelConfig: React.FC<Props> = ({ layers, setLayers }) => {
  // Input: 13x13 axial grid = 127 hexes (radius 6)
  // Output: 2 moves, each a softmax over 127 hexes = 254 nodes
  const INPUT_NODES = 127; 
  const OUTPUT_NODES = 254; 

  const updateLayer = (index: number, val: number) => {
    const newLayers = [...layers];
    newLayers[index] = Math.max(1, Math.min(512, val ?? 1));
    setLayers(newLayers);
  };

  const addLayer = () => {
    if (layers.length < 10) {
      setLayers([...layers, 64]);
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
        <p className="section-desc">Configure the deep learning model layers. Input and Output are fixed by the game environment.</p>
      </div>
      
      <div className="config-layout">
        <section className="fixed-layers card">
          <div className="layer-type-group">
            <h3>Input Layer</h3>
            <div className="fixed-node-badge">{INPUT_NODES} Nodes</div>
            <p>13x13 Local Board Crop (RingIndex A0 - F35)</p>
          </div>
          
          <div className="layer-type-group" style={{ marginTop: '30px' }}>
            <h3>Output Layer</h3>
            <div className="fixed-node-badge">{OUTPUT_NODES} Nodes</div>
            <p>Two moves in RingIndex (2 &times; 127 policy head)</p>
          </div>
        </section>

        <section className="layer-controls card">
          <h3>Hidden Layers (Adjustable)</h3>
          <p className="section-desc">Add or tune the processing depth of the AI's brain.</p>
          
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
            <button className="add-layer-btn" onClick={addLayer} disabled={layers.length >= 10}>
              + Add Hidden Layer
            </button>
          </div>
          
          <div className="complexity-stats">
            <div className="stat-row">
              <span>Total Hidden Nodes:</span>
              <strong>{layers.reduce((a, b) => a + b, 0)}</strong>
            </div>
            <div className="stat-row">
              <span>Total Network Depth:</span>
              <strong>{layers.length + 2} Layers</strong>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
