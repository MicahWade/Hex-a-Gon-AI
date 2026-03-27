import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { createModel } from '../ai/modelBuilder';
import { Trainer } from '../ai/trainer';
import type { TrainingConfig } from '../ai/trainer';
import type { BoardState, Coord, Player } from '../types';
import { getVaultMetadata, saveModelToVault, loadModelFromVault, deleteModelFromVault } from '../ai/modelVault';
import type { ModelMetadata } from '../ai/modelVault';

interface Props {
  isTraining: boolean;
  setIsTraining: (val: boolean) => void;
  layers: number[];
}

export const AITraining: React.FC<Props> = ({ isTraining, setIsTraining, layers }) => {
  const [generations, setGenerations] = useState(0);
  const [loss, setLoss] = useState<number>(0);
  const [logs, setLog] = useState<string[]>(["[System] Ready for training."]);
  const [vault, setVault] = useState<ModelMetadata[]>([]);
  const [currentModelName, setCurrentModelName] = useState<string>("default-model");
  const [isChampionship, setIsChampionship] = useState(false);
  const [champResults, setChampResults] = useState<{ p1: number, p2: number } | null>(null);

  const [rewards, setRewards] = useState({
    p1Win: 1.0,
    p2Win: 1.1,
    p1Draw: 0.4,
    p2Draw: 0.6,
    threat: 0.02,
    efficiency: -0.005
  });

  const trainerRef = useRef<Trainer | null>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);

  useEffect(() => {
    setVault(getVaultMetadata());
  }, []);

  const addLog = (msg: string) => {
    setLog(prev => [msg, ...prev].slice(0, 50));
  };

  const initTrainer = (model: tf.LayersModel) => {
    modelRef.current = model;
    trainerRef.current = new Trainer(model);
  };

  const handleCreateNew = () => {
    const focalRadius = 14;
    const selfRadius = 8;
    const memoryRadius = 6;
    const HEX_INPUTS = (3 * focalRadius * (focalRadius + 1) + 1) + 
                      (3 * selfRadius * (selfRadius + 1) + 1) + 
                      (3 * memoryRadius * (memoryRadius + 1) + 1) * 4;
    const INPUT_NODES = HEX_INPUTS + 3 + 12;
    const OUTPUT_NODES = HEX_INPUTS;

    const model = createModel({
      inputNodes: INPUT_NODES,
      outputNodes: OUTPUT_NODES,
      hiddenLayers: layers,
      learningRate: 0.001
    });
    initTrainer(model);
    addLog(`[System] New model initialized with ${layers.length} hidden layers.`);
  };

  const handleSave = async () => {
    if (!modelRef.current) return;
    const name = prompt("Enter model name:", currentModelName) || currentModelName;
    
    // Metadata for verification
    const focalRadius = 14;
    const selfRadius = 8;
    const memoryRadius = 6;
    const HEX_INPUTS = (3 * focalRadius * (focalRadius + 1) + 1) + 
                      (3 * selfRadius * (selfRadius + 1) + 1) + 
                      (3 * memoryRadius * (memoryRadius + 1) + 1) * 4;

    const meta: ModelMetadata = {
      name,
      timestamp: Date.now(),
      inputNodes: HEX_INPUTS + 3 + 12,
      outputNodes: HEX_INPUTS,
      hiddenLayers: layers
    };

    await saveModelToVault(modelRef.current, meta);
    setCurrentModelName(name);
    setVault(getVaultMetadata());
    addLog(`[System] Model '${name}' saved to vault.`);
  };

  const handleLoad = async (name: string) => {
    try {
      const model = await loadModelFromVault(name);
      initTrainer(model);
      setCurrentModelName(name);
      addLog(`[System] Model '${name}' loaded successfully.`);
    } catch (e) {
      addLog(`[Error] Failed to load '${name}'.`);
    }
  };

  const handleDelete = async (name: string) => {
    if (confirm(`Delete model '${name}'?`)) {
      await deleteModelFromVault(name);
      setVault(getVaultMetadata());
    }
  };

  const runChampionship = async () => {
    if (!modelRef.current || vault.length === 0) return;
    const opponentName = prompt("Enter model name to play against:", vault[0].name);
    if (!opponentName) return;

    setIsChampionship(true);
    addLog(`[Champ] Starting Championship vs ${opponentName}...`);
    
    try {
      const opponentModel = await loadModelFromVault(opponentName);
      let p1Wins = 0;
      let p2Wins = 0;
      const config: TrainingConfig = {
        learningRate: 0.001,
        batchSize: 32,
        gamma: 0.95,
        epsilon: 0, // No randomness for champ matches
        rewards
      };
      const radii = { global: 14, self: 8, memory: 6 };

      // 10 Games total, 5 as P1, 5 as P2
      for (let g = 0; g < 10; g++) {

        let board: BoardState = new Map();
        let currentPlayer: Player = 1;
        let winner: Player | null = null;
        let foci: Coord[] = Array(6).fill({ q: 0, r: 0 });
        const modelIsP1 = g < 5;

        while (!winner && board.size < 100) {
          const m = (currentPlayer === 1 && modelIsP1) || (currentPlayer === 2 && !modelIsP1) 
            ? modelRef.current : opponentModel;
          
          const result = await trainerRef.current!.playTurn(board, currentPlayer, foci, radii, config, m);
          board = result.board;
          winner = result.winner;
          if (result.moves.length > 0) foci[0] = result.moves[result.moves.length - 1];
          currentPlayer = currentPlayer === 1 ? 2 : 1;
        }

        if (winner) {
          if ((winner === 1 && modelIsP1) || (winner === 2 && !modelIsP1)) p1Wins++;
          else p2Wins++;
        }
        setChampResults({ p1: p1Wins, p2: p2Wins });
      }
      addLog(`[Champ] Final Score: Current ${p1Wins} - ${p2Wins} ${opponentName}`);
    } catch (e) {
      addLog("[Error] Championship failed.");
    }
    setIsChampionship(false);
  };

  // Training Loop Effect
  useEffect(() => {
    let active = true;
    if (!isTraining || !trainerRef.current) return;

    const runCycle = async () => {
      const config: TrainingConfig = { learningRate: 0.001, batchSize: 32, gamma: 0.95, epsilon: 0.2, rewards };
      const radii = { global: 14, self: 8, memory: 6 };
      let board: BoardState = new Map();
      let currentPlayer: Player = 1;
      let winner: Player | null = null;
      let foci: Coord[] = Array(6).fill({ q: 0, r: 0 });
      let turns = 0;

      while (!winner && turns < 100 && active && isTraining) {
        const result = await trainerRef.current!.playTurn(board, currentPlayer, foci, radii, config);
        board = result.board;
        winner = result.winner;
        if (result.moves.length > 0) foci[0] = result.moves[result.moves.length - 1];
        currentPlayer = currentPlayer === 1 ? 2 : 1;
        turns++;

        if (turns % 10 === 0) {
          const l = await trainerRef.current!.trainBatch(32);
          if (l) setLoss(l);
        }
      }
      setGenerations(g => g + 1);
      if (active && isTraining) setTimeout(runCycle, 50);
    };
    runCycle();
    return () => { active = false; };
  }, [isTraining, rewards]);

  return (
    <div className="tab-content ai-view">
      <div className="settings-header">
        <h2>AI Training Lab</h2>
        <div className="current-model-info">
          <span className="model-badge">Active Model: {currentModelName}</span>
        </div>
      </div>
      
      <div className="ai-grid">
        <div className="ai-main-col">
          <section className="model-vault card">
            <h3>Model Vault</h3>
            <div className="vault-list">
              {vault.length === 0 && <p className="no-moves">No saved models.</p>}
              {vault.map(m => (
                <div key={m.name} className="vault-item">
                  <div className="vault-info">
                    <strong>{m.name}</strong>
                    <span>{new Date(m.timestamp).toLocaleDateString()}</span>
                  </div>
                  <div className="vault-actions">
                    <button onClick={() => handleLoad(m.name)}>Load</button>
                    <button className="delete-btn" onClick={() => handleDelete(m.name)}>&times;</button>
                  </div>
                </div>
              ))}
            </div>
            <button className="add-layer-btn" style={{marginTop: '15px'}} onClick={handleCreateNew}>Initialize New Model</button>
          </section>

          <section className="training-log card">
            <h3>System Logs</h3>
            <div className="log-window">
              {logs.map((log, i) => <p key={i}>{log}</p>)}
            </div>
          </section>

          <section className="reward-config card">
            <h3>Reward System</h3>
            <p className="section-desc">Adjust training incentives.</p>
            <div className="reward-grid">
              <div className="input-group">
                <label>Win P1/P2</label>
                <div style={{display: 'flex', gap: '10px'}}>
                  <input type="number" value={rewards.p1Win} step={0.1} onChange={e => setRewards({...rewards, p1Win: parseFloat(e.target.value)})} />
                  <input type="number" value={rewards.p2Win} step={0.1} onChange={e => setRewards({...rewards, p2Win: parseFloat(e.target.value)})} />
                </div>
              </div>
              <div className="input-group">
                <label>Draw P1/P2</label>
                <div style={{display: 'flex', gap: '10px'}}>
                  <input type="number" value={rewards.p1Draw} step={0.1} onChange={e => setRewards({...rewards, p1Draw: parseFloat(e.target.value)})} />
                  <input type="number" value={rewards.p2Draw} step={0.1} onChange={e => setRewards({...rewards, p2Draw: parseFloat(e.target.value)})} />
                </div>
              </div>
              <div className="input-group">
                <label>Threat Detection</label>
                <input type="number" value={rewards.threat} step={0.01} onChange={e => setRewards({...rewards, threat: parseFloat(e.target.value)})} />
              </div>
              <div className="input-group">
                <label>Efficiency</label>
                <input type="number" value={rewards.efficiency} step={0.001} onChange={e => setRewards({...rewards, efficiency: parseFloat(e.target.value)})} />
              </div>
            </div>
          </section>
        </div>

        <div className="ai-side-col">
          <section className="training-stats card">
            <h3>Controls</h3>
            <div className="actions" style={{ flexDirection: 'column', gap: '10px' }}>
              <button className={isTraining ? 'stop-btn' : 'start-btn'} onClick={() => setIsTraining(!isTraining)}>
                {isTraining ? 'Stop Training' : 'Start Training'}
              </button>
              <button className="reset-btn" onClick={handleSave}>Save to Vault</button>
              <button 
                className="recommend-btn" 
                style={{width: '100%', height: '40px', fontSize: '14px'}}
                onClick={runChampionship}
                disabled={isTraining || isChampionship}
              >
                {isChampionship ? 'Champ Match...' : 'Run Championship Match'}
              </button>
            </div>

            {champResults && (
              <div className="champ-box" style={{marginTop: '15px', background: 'rgba(0,0,0,0.3)', padding: '10px', borderRadius: '8px'}}>
                <h4 style={{margin: '0 0 5px 0', fontSize: '12px', color: 'var(--accent-color)'}}>Championship Score</h4>
                <div style={{display: 'flex', justifyContent: 'space-around', fontWeight: 'bold'}}>
                  <span>Current: {champResults.p1}</span>
                  <span>vs</span>
                  <span>Opponent: {champResults.p2}</span>
                </div>
              </div>
            )}

            <div className="stat-summary" style={{ marginTop: '20px' }}>
              <div className="stat-card"><span>Gen:</span> <span>{generations}</span></div>
              <div className="stat-card"><span>Loss:</span> <span>{loss.toFixed(4)}</span></div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
