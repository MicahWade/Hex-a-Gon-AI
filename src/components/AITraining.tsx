import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { createModel } from '../ai/modelBuilder';
import { Trainer } from '../ai/trainer';
import type { TrainingConfig } from '../ai/trainer';
import type { BoardState, Coord, Player } from '../types';
import { getVaultMetadata, saveModelToVault, loadModelFromVault, deleteModelFromVault } from '../ai/modelVault';
import type { ModelMetadata } from '../ai/modelVault';
import { encodeState } from '../ai/encoder';

interface Props {
  isTraining: boolean;
  setIsTraining: (val: boolean) => void;
  layers: number[];
  setLayers: (newLayers: number[]) => void;
  focalRadii: { global: number; self: number; memory: number };
  setFocalRadii: React.Dispatch<React.SetStateAction<{ global: number; self: number; memory: number }>>;
  generations: number;
  setGenerations: React.Dispatch<React.SetStateAction<number>>;
  loss: number;
  setLoss: React.Dispatch<React.SetStateAction<number>>;
}

export const AITraining: React.FC<Props> = ({ 
  isTraining, setIsTraining, layers, setLayers, focalRadii, setFocalRadii,
  generations, setGenerations, loss, setLoss
}) => {
  const [logs, setLog] = useState<string[]>(["[System] Ready for training."]);
  const [vault, setVault] = useState<ModelMetadata[]>([]);
  const [currentModelName, setCurrentModelName] = useState<string>("default-model");
  const [isChampionship, setIsChampionship] = useState(false);
  const [champResults, setChampResults] = useState<{ p1: number, p2: number } | null>(null);
  const [maxTurns, setMaxTurns] = useState(100);
  const [batchSize, setBatchSize] = useState(64);
  const [epsilon, setEpsilon] = useState(0.2);

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

  const parseSafeFloat = (val: string): number => {
    const parsed = parseFloat(val);
    return isNaN(parsed) ? 0 : parsed;
  };

  const initTrainer = (model: tf.LayersModel) => {
    if (!model.optimizer) {
      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
    }
    modelRef.current = model;
    trainerRef.current = new Trainer(model);
  };

  const getIOConfig = () => {
    const hexInputs = (3 * focalRadii.global * (focalRadii.global + 1) + 1) + 
                      (3 * focalRadii.self * (focalRadii.self + 1) + 1) + 
                      (3 * focalRadii.memory * (focalRadii.memory + 1) + 1) * 4;
    return {
      inputNodes: hexInputs + 4 + 12,
      outputNodes: hexInputs
    };
  };

  const handleCreateNew = () => {
    const { inputNodes, outputNodes } = getIOConfig();
    const model = createModel({
      inputNodes,
      outputNodes,
      hiddenLayers: layers,
      learningRate: 0.001
    });
    initTrainer(model);
    setGenerations(0);
    setLoss(0);
    addLog(`[System] New model: ${inputNodes} in -> [${layers.join(',')}] -> ${outputNodes} out`);
  };

  const toggleTraining = async () => {
    if (!isTraining) {
      if (!modelRef.current) {
        handleCreateNew();
      }
      addLog("[System] Training resumed.");
    } else {
      addLog("[System] Training paused.");
    }
    setIsTraining(!isTraining);
  };

  const handleSave = async () => {
    if (!modelRef.current) return;
    const name = prompt("Enter model name:", currentModelName) || currentModelName;
    const { inputNodes, outputNodes } = getIOConfig();

    const meta: ModelMetadata = {
      name,
      timestamp: Date.now(),
      inputNodes,
      outputNodes,
      hiddenLayers: layers,
      focalRadii: { ...focalRadii },
      generation: generations,
      maxTurns,
      batchSize,
      epsilon
    };

    await saveModelToVault(modelRef.current, meta);
    setCurrentModelName(name);
    setVault(getVaultMetadata());
    addLog(`[System] Saved '${name}' (${inputNodes} in, [${layers.join(',')}] hidden, ${outputNodes} out)`);
  };

  const handleLoad = async (name: string) => {
    try {
      const model = await loadModelFromVault(name);
      const meta = vault.find(m => m.name === name);
      if (meta) {
        setLayers(meta.hiddenLayers);
        if (meta.focalRadii) setFocalRadii(meta.focalRadii);
        if (meta.generation !== undefined) setGenerations(meta.generation);
        if (meta.maxTurns !== undefined) setMaxTurns(meta.maxTurns);
        if (meta.batchSize !== undefined) setBatchSize(meta.batchSize);
        if (meta.epsilon !== undefined) setEpsilon(meta.epsilon);
        addLog(`[System] Synced Architecture, Stats & Settings for '${name}'.`);
      }
      initTrainer(model);
      if (trainerRef.current) trainerRef.current.clearMemory();
      setCurrentModelName(name);
      addLog(`[System] Loaded '${name}' (${meta?.inputNodes} in, [${meta?.hiddenLayers.join(',')}] hidden, ${meta?.outputNodes} out)`);
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
    const opponentName = prompt("Enter opponent model name:", vault[0].name);
    if (!opponentName) return;

    setIsChampionship(true);
    addLog(`[Champ] Battle: ${currentModelName} vs ${opponentName}`);
    
    try {
      const opponentModel = await loadModelFromVault(opponentName);
      const opponentMeta = vault.find(m => m.name === opponentName);
      
      if (opponentMeta && (
        opponentMeta.focalRadii.global !== focalRadii.global ||
        opponentMeta.focalRadii.self !== focalRadii.self ||
        opponentMeta.focalRadii.memory !== focalRadii.memory
      )) {
        addLog("[Error] Opponent vision radii do not match current model. Championship cancelled.");
        setIsChampionship(false);
        return;
      }

      let p1Wins = 0;
      let p2Wins = 0;
      const config: TrainingConfig = { learningRate: 0.001, batchSize: 64, gamma: 0.95, epsilon: 0, rewards };

      for (let g = 0; g < 10; g++) {
        let board: BoardState = new Map();
        let currentPlayer: Player = 1;
        let winner: Player | null = null;
        let foci: Coord[] = Array(6).fill({ q: 0, r: 0 });
        const modelIsP1 = g < 5;
        let turns = 0;

        while (!winner && board.size < maxTurns * 2) {
          const m = (currentPlayer === 1 && modelIsP1) || (currentPlayer === 2 && !modelIsP1) 
            ? modelRef.current : opponentModel;
          
          const result = await trainerRef.current!.playTurn(board, currentPlayer, foci, focalRadii, config, turns, maxTurns, m);
          board = result.board;
          winner = result.winner;
          if (result.moves.length > 0) foci[0] = result.moves[result.moves.length - 1];
          currentPlayer = currentPlayer === 1 ? 2 : 1;
          turns++;
        }

        if (winner) {
          if ((winner === 1 && modelIsP1) || (winner === 2 && !modelIsP1)) p1Wins++;
          else p2Wins++;
        }
        setChampResults({ p1: p1Wins, p2: p2Wins });
      }
    } catch (e) {
      addLog("[Error] Champ match failed.");
    }
    setIsChampionship(false);
  };

  useEffect(() => {
    let active = true;
    if (!isTraining || !trainerRef.current) return;

    const runCycle = async () => {
      const config: TrainingConfig = { 
        learningRate: 0.001, batchSize: 64, gamma: 0.95, epsilon: epsilon, rewards 
      };
      let board: BoardState = new Map();
      let currentPlayer: Player = 1;
      let winner: Player | null = null;
      let foci: Coord[] = Array(6).fill({ q: 0, r: 0 });
      let turns = 0;

      const gameHistory: { state: number[], action: number, player: Player }[] = [];

      while (!winner && turns < maxTurns && active && isTraining) {
        const stateBefore = encodeState(board, currentPlayer, foci, focalRadii, turns, maxTurns);
        const result = await trainerRef.current!.playTurn(board, currentPlayer, foci, focalRadii, config, turns, maxTurns);
        
        result.actionIndices.forEach(actionIdx => {
          gameHistory.push({ state: stateBefore, action: actionIdx, player: currentPlayer });
        });

        board = result.board;
        winner = result.winner;
        if (result.moves.length > 0) foci[0] = result.moves[result.moves.length - 1];
        currentPlayer = currentPlayer === 1 ? 2 : 1;
        turns++;

        if (turns % 5 === 0) {
          const l = await trainerRef.current!.trainBatch(batchSize);
          if (l) setLoss(l);
        }
      }

      gameHistory.forEach(exp => {
        let reward = rewards.efficiency;
        if (winner) {
          if (exp.player === winner) reward += (winner === 1 ? rewards.p1Win : rewards.p2Win);
          else reward -= 1.0;
        } else {
          reward += (exp.player === 1 ? rewards.p1Draw : rewards.p2Draw);
        }
        trainerRef.current!.addToMemory(exp.state, exp.action, reward, null);
      });

      if (winner) addLog(`[Game] P${winner} won in ${turns} turns.`);
      setGenerations(g => g + 1);
      
      if (active && isTraining) setTimeout(runCycle, 10);
    };
    runCycle();
    return () => { active = false; };
  }, [isTraining, rewards, maxTurns, focalRadii, epsilon, batchSize, setGenerations, setLoss]);

  return (
    <div className="tab-content ai-view">
      <div className="settings-header">
        <h2>AI Training Lab</h2>
        <div className="current-model-info">
          <span className="model-badge">Active Model: {currentModelName}</span>
        </div>
      </div>

      <section className="ai-top-bar card">
        <div className="top-bar-layout">
          <div className="control-section">
            <div className="mini-input-row">
              <label>Max Turns</label>
              <input type="number" value={maxTurns || 0} onChange={e => setMaxTurns(parseSafeFloat(e.target.value))} min="10" max="500" />
            </div>
            <div className="mini-input-row">
              <label>Batch Size</label>
              <input type="number" value={batchSize || 0} onChange={e => setBatchSize(parseSafeFloat(e.target.value))} step={32} min="32" max="512" />
            </div>
            <div className="mini-input-row">
              <label>Randomness</label>
              <input type="number" value={epsilon} onChange={e => setEpsilon(parseSafeFloat(e.target.value))} step={0.05} min="0" max="1" />
            </div>
            <div className="action-buttons">
              <button className={isTraining ? 'stop-btn' : 'start-btn'} onClick={toggleTraining}>
                {isTraining ? 'Stop Training' : 'Start Training'}
              </button>
              <button className="reset-btn" onClick={handleSave}>Save to Vault</button>
              <button className="recommend-btn" onClick={runChampionship} disabled={isTraining || isChampionship}>
                {isChampionship ? 'Match...' : 'Championship'}
              </button>
            </div>
          </div>

          <div className="stats-section">
            <div className="stat-pill"><span>Gen</span> <strong>{generations}</strong></div>
            <div className="stat-pill"><span>Loss</span> <strong>{loss.toFixed(4)}</strong></div>
            {champResults && (
              <div className="champ-pill">
                <span>Champ</span> <strong>{champResults.p1} - {champResults.p2}</strong>
              </div>
            )}
          </div>
        </div>
      </section>
      
      <div className="ai-grid">
        <div className="ai-main-col">
          <section className="model-vault card full-height-card">
            <h3>Model Vault</h3>
            <div className="vault-list">
              {vault.length === 0 && <p className="no-moves">No saved models.</p>}
              {vault.map(m => (
                <div key={m.name} className="vault-item">
                  <div className="vault-info">
                    <strong>{m.name}</strong>
                    <span>{new Date(m.timestamp).toLocaleDateString()} • Gen {m.generation}</span>
                  </div>
                  <div className="vault-actions">
                    <button onClick={() => handleLoad(m.name)}>Load</button>
                    <button className="delete-btn" onClick={() => handleDelete(m.name)}>&times;</button>
                  </div>
                </div>
              ))}
            </div>
            <button className="add-layer-btn" style={{marginTop: 'auto'}} onClick={handleCreateNew}>Initialize New Model</button>
          </section>
        </div>

        <div className="ai-side-col">
          <section className="reward-config card full-height-card">
            <h3>Reward System</h3>
            <p className="section-desc">Adjust incentives.</p>
            <div className="reward-grid">
              <div className="input-group">
                <label>Win P1 / P2</label>
                <div className="dual-input">
                  <input type="number" value={rewards.p1Win || 0} step={0.1} onChange={e => setRewards({...rewards, p1Win: parseSafeFloat(e.target.value)})} />
                  <input type="number" value={rewards.p2Win || 0} step={0.1} onChange={e => setRewards({...rewards, p2Win: parseSafeFloat(e.target.value)})} />
                </div>
              </div>
              <div className="input-group">
                <label>Draw P1 / P2</label>
                <div className="dual-input">
                  <input type="number" value={rewards.p1Draw || 0} step={0.1} onChange={e => setRewards({...rewards, p1Draw: parseSafeFloat(e.target.value)})} />
                  <input type="number" value={rewards.p2Draw || 0} step={0.1} onChange={e => setRewards({...rewards, p2Draw: parseSafeFloat(e.target.value)})} />
                </div>
              </div>
              <div className="input-group">
                <label>Threat Detection</label>
                <input type="number" value={rewards.threat || 0} step={0.01} onChange={e => setRewards({...rewards, threat: parseSafeFloat(e.target.value)})} />
              </div>
              <div className="input-group">
                <label>Efficiency</label>
                <input type="number" value={rewards.efficiency || 0} step={0.001} onChange={e => setRewards({...rewards, efficiency: parseSafeFloat(e.target.value)})} />
              </div>
            </div>
          </section>
        </div>
      </div>

      <section className="training-log card" style={{marginTop: '20px'}}>
        <h3>System Logs</h3>
        <div className="log-window" style={{height: '200px'}}>
          {logs.map((log, i) => <p key={i}>{log}</p>)}
        </div>
      </section>
    </div>
  );
};
