import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { createModel } from '../ai/modelBuilder';
import { Trainer } from '../ai/trainer';
import type { TrainingConfig } from '../ai/trainer';
import type { BoardState, Coord, Player } from '../types';
import { getVaultMetadata, saveModelToVault, loadModelFromVault, deleteModelFromVault } from '../ai/modelVault';
import type { ModelMetadata } from '../ai/modelVault';
import { encodeState } from '../ai/encoder';
import { getMaxLine } from '../gameLogic';

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
  currentModelName: string;
  setCurrentModelName: (name: string) => void;
}

export const AITraining: React.FC<Props> = ({ 
  isTraining, setIsTraining, layers, setLayers, focalRadii, setFocalRadii,
  generations, setGenerations, loss, setLoss,
  currentModelName, setCurrentModelName
}) => {
  const [logs, setLog] = useState<string[]>(["[System] Ready for training."]);
  const [vault, setVault] = useState<ModelMetadata[]>([]);
  const [isChampionship, setIsChampionship] = useState(false);
  const [champResults, setChampResults] = useState<{ p1: number, p2: number } | null>(null);
  const [maxTurns, setMaxTurns] = useState(100);
  const [batchSize, setBatchSize] = useState(64);
  const [epsilon, setEpsilon] = useState(0.2);
  const [autoSaveFreq, setAutoSaveFreq] = useState(10);

  const [rewards, setRewards] = useState({
    p1Win: 2.0,
    p2Win: 2.2,
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
    addLog(`[System] New model initialized.`);
  };

  const toggleTraining = async () => {
    if (!isTraining) {
      if (!modelRef.current) handleCreateNew();
      addLog("[System] Training resumed.");
    } else {
      addLog("[System] Training paused.");
    }
    setIsTraining(!isTraining);
  };

  const performSave = async (isAuto = false) => {
    if (!modelRef.current) return;
    const name = isAuto ? currentModelName : (prompt("Enter model name:", currentModelName) || currentModelName);
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
    if (!isAuto) setCurrentModelName(name);
    setVault(getVaultMetadata());
    addLog(`[System] Model '${name}' saved ${isAuto ? '(Auto)' : ''}.`);
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
      }
      initTrainer(model);
      if (trainerRef.current) trainerRef.current.clearMemory();
      setCurrentModelName(name);
      addLog(`[System] Model '${name}' loaded.`);
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
        addLog("[Error] Opponent vision radii mismatch.");
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
        let turns = 0;
        const modelIsP1 = g < 5;

        while (!winner && board.size < maxTurns * 2) {
          const m = (currentPlayer === 1 && modelIsP1) || (currentPlayer === 2 && !modelIsP1) ? modelRef.current : opponentModel;
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
      addLog("[Error] Champ failed.");
    }
    setIsChampionship(false);
  };

  useEffect(() => {
    let active = true;
    if (!isTraining || !trainerRef.current) return;

    const runCycle = async () => {
      try {
        const config: TrainingConfig = { learningRate: 0.001, batchSize, gamma: 0.95, epsilon, rewards };
        let board: BoardState = new Map();
        let currentPlayer: Player = 1;
        let winner: Player | null = null;
        let foci: Coord[] = Array(6).fill({ q: 0, r: 0 });
        let turns = 0;
        const gameHistory: { state: number[], action: number, player: Player, turn: number, isThreat: boolean }[] = [];

        while (!winner && turns < maxTurns && active && isTraining) {
          const result = await tf.tidy(() => {
            const stateBefore = encodeState(board, currentPlayer, foci, focalRadii, turns, maxTurns);
            return {
              stateBefore,
              playResult: trainerRef.current!.playTurn(board, currentPlayer, foci, focalRadii, config, turns, maxTurns)
            };
          });

          const { stateBefore, playResult } = result;
          const actualResult = await playResult;
          
          actualResult.moves.forEach((move, i) => {
            const isThreat = getMaxLine(board, move.q, move.r, currentPlayer) >= 4;
            gameHistory.push({ state: stateBefore, action: actualResult.actionIndices[i], player: currentPlayer, turn: turns, isThreat });
          });

          board = actualResult.board;
          winner = actualResult.winner;
          if (actualResult.moves.length > 0) foci[0] = actualResult.moves[actualResult.moves.length - 1];
          currentPlayer = currentPlayer === 1 ? 2 : 1;
          turns++;
        }

        const playerResults = [1, 2].map(p => {
          const pExps = gameHistory.filter(exp => exp.player === p);
          const base = winner ? (winner === p ? (p === 1 ? rewards.p1Win : rewards.p2Win) : -1.0) : (p === 1 ? rewards.p1Draw : rewards.p2Draw);
          let bonus = 0;
          pExps.forEach(exp => {
            bonus += rewards.efficiency;
            if (exp.isThreat) bonus += rewards.threat * (1.0 - (exp.turn / maxTurns) * 0.5);
          });
          const cap = Math.abs(base) * 0.5;
          return { player: p, experiences: pExps, total: base + Math.max(-cap, Math.min(cap, bonus)) };
        });

        if (winner) {
          const winIdx = playerResults.findIndex(r => r.player === winner);
          const loseIdx = playerResults.findIndex(r => r.player !== winner);
          if (playerResults[winIdx].total <= playerResults[loseIdx].total) playerResults[winIdx].total = playerResults[loseIdx].total + 0.1;
        }

        playerResults.forEach(res => {
          res.experiences.forEach(exp => trainerRef.current!.addToMemory(exp.state, exp.action, res.total, null));
        });

        const l = await trainerRef.current!.trainBatch(batchSize);
        if (l) setLoss(l);

        const nextGen = generations + 1;
        setGenerations(nextGen);
        if (nextGen > 0 && nextGen % autoSaveFreq === 0) performSave(true);

        if (winner) addLog(`[Game] P${winner} won in ${turns} turns.`);
        tf.engine().startScope();
        tf.engine().endScope(); 
        
        if (active && isTraining) setTimeout(runCycle, 150);
      } catch (err) {
        addLog(`[Memory Shield] Purging tensors and restarting...`);
        tf.engine().disposeVariables();
        if (active && isTraining) setTimeout(runCycle, 2000);
      }
    };
    runCycle();
    return () => { active = false; };
  }, [isTraining, rewards, maxTurns, focalRadii, epsilon, batchSize, autoSaveFreq, generations, setGenerations, setLoss]);

  return (
    <div className="tab-content ai-view">
      <div className="settings-header">
        <h2>AI Training Lab</h2>
      </div>

      <section className="ai-top-bar card">
        {/* ROW 1: INPUTS */}
        <div className="top-bar-row">
          <div className="input-group-horizontal">
            <div className="mini-input-row"><label>Max Turns</label><input type="number" value={maxTurns || 0} onChange={e => setMaxTurns(parseSafeFloat(e.target.value))} min="10" max="500" /></div>
            <div className="mini-input-row"><label>Batch Size</label><input type="number" value={batchSize || 0} onChange={e => setBatchSize(parseSafeFloat(e.target.value))} step={32} min="32" max="512" /></div>
            <div className="mini-input-row"><label>Randomness</label><input type="number" value={epsilon} onChange={e => setEpsilon(parseSafeFloat(e.target.value))} step={0.05} min="0" max="1" /></div>
            <div className="mini-input-row"><label>Auto-Save</label><input type="number" value={autoSaveFreq} onChange={e => setAutoSaveFreq(Math.max(1, parseInt(e.target.value)))} min="1" max="1000" /></div>
          </div>
        </div>

        {/* ROW 2: ACTIONS & STATS */}
        <div className="top-bar-row" style={{marginTop: '15px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '15px'}}>
          <div className="action-buttons">
            <button className={isTraining ? 'stop-btn' : 'start-btn'} onClick={toggleTraining}>{isTraining ? 'Stop' : 'Start Training'}</button>
            <button className="reset-btn" onClick={() => performSave(false)}>Save</button>
            <button className="recommend-btn" onClick={runChampionship} disabled={isTraining || isChampionship}>Champ</button>
          </div>
          <div className="stats-section">
            <div className="stat-pill model-name-pill"><span>Model</span> <strong>{currentModelName}</strong></div>
            <div className="stat-pill"><span>Gen</span> <strong>{generations}</strong></div>
            <div className="stat-pill"><span>Loss</span> <strong>{loss.toFixed(4)}</strong></div>
            {champResults && <div className="champ-pill"><span>Score</span> <strong>{champResults.p1}-{champResults.p2}</strong></div>}
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
                  <div className="vault-info"><strong>{m.name}</strong><span>{new Date(m.timestamp).toLocaleDateString()} • Gen {m.generation}</span></div>
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
              <div className="input-group"><label>Win P1 / P2</label><div className="dual-input"><input type="number" value={rewards.p1Win || 0} step={0.1} onChange={e => setRewards({...rewards, p1Win: parseSafeFloat(e.target.value)})} /><input type="number" value={rewards.p2Win || 0} step={0.1} onChange={e => setRewards({...rewards, p2Win: parseSafeFloat(e.target.value)})} /></div></div>
              <div className="input-group"><label>Draw P1 / P2</label><div className="dual-input"><input type="number" value={rewards.p1Draw || 0} step={0.1} onChange={e => setRewards({...rewards, p1Draw: parseSafeFloat(e.target.value)})} /><input type="number" value={rewards.p2Draw || 0} step={0.1} onChange={e => setRewards({...rewards, p2Draw: parseSafeFloat(e.target.value)})} /></div></div>
              <div className="input-group"><label>Threat Detection</label><input type="number" value={rewards.threat || 0} step={0.01} onChange={e => setRewards({...rewards, threat: parseSafeFloat(e.target.value)})} /></div>
              <div className="input-group"><label>Efficiency</label><input type="number" value={rewards.efficiency || 0} step={0.001} onChange={e => setRewards({...rewards, efficiency: parseSafeFloat(e.target.value)})} /></div>
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
