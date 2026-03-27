import React, { useState } from 'react';

interface Props {
  layers: number[];
  setLayers: (newLayers: number[]) => void;
}

export const ModelConfig: React.FC<Props> = ({ layers, setLayers }) => {
  const [maxRadius, setMaxRadius] = useState(20);
  const [seqLength, setSeqLength] = useState(100);

  // Input: Sequence of pieces [Ring, Index, Player]
  const INPUT_NODES = seqLength * 3; 
  // Output: (Ring selection + Index selection) * 2 moves
  const RING_NODES = maxRadius + 1;
  const INDEX_NODES = maxRadius * 6;
  const OUTPUT_NODES = (RING_NODES + INDEX_NODES) * 2; 

  const updateLayer = (index: number, val: number) => {
    const newLayers = [...layers];
    newLayers[index] = Math.max(1, Math.min(1024, val ?? 1));
    setLayers(newLayers);
  };

  const addLayer = () => {
    if (layers.length < 12) {
      setLayers([...layers, 128]);
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
        <p className="section-desc">Entity-based Encoding for large-range infinite board support.</p>
      </div>
      
      <div className="config-layout">
        <section className="fixed-layers card">
          <div className="layer-type-group">
            <h3>Input: Entity Sequence</h3>
            <div className="fixed-node-badge">{INPUT_NODES} Nodes</div>
            <p>List of the last {seqLength} pieces placed.</p>
            <div className="mini-input">
              <label>Sequence Length</label>
              <input type="number" value={seqLength} onChange={e => setSeqLength(parseInt(e.target.value))} />
            </div>
          </div>
          
          <div className="layer-type-group" style={{ marginTop: '30px' }}>
            <h3>Output: Ring-Index Dual Head</h3>
            <div className="fixed-node-badge">{OUTPUT_NODES} Nodes</div>
            <p>Predicts [Ring] and [Index] for Move 1 & Move 2.</p>
            <div className="mini-input">
              <label>Max Ring Radius</label>
              <input type="number" value={maxRadius} onChange={e => setMaxRadius(parseInt(e.target.value))} />
            </div>
          </div>
        </section>

        <section className="layer-controls card">
          <h3>Hidden Layers (Adjustable)</h3>
          <p className="section-desc">Neurons per layer. Larger layers handle more complex strategies.</p>
          
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
            <div className="stat-row">
              <span>Network Depth:</span>
              <strong>{layers.length + 2} Layers</strong>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
