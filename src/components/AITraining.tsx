import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { createModel } from '../ai/modelBuilder';
import { Trainer } from '../ai/trainer';
import type { TrainingConfig } from '../ai/trainer';
import type { BoardState, Coord, Player } from '../types';
import { getVaultMetadata, saveModelToVault, loadModelFromVault, deleteModelFromVault } from '../ai/modelVault';
import type { ModelMetadata } from '../ai/modelVault';
import { encodeState, coordToIndex, decodeMove } from '../ai/encoder';
import { getMaxLine, rotateBoard, rotateCoord, coordToString } from '../gameLogic';

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
  trainerRef: React.MutableRefObject<Trainer | null>;
  modelRef: React.MutableRefObject<tf.LayersModel | null>;
  setIsAiLoaded: (val: boolean) => void;
}

interface GameSession {
  board: BoardState;
  currentPlayer: Player;
  winner: Player | null;
  foci: Coord[];
  turns: number;
  history: { boardBefore: BoardState, foci: Coord[], move: Coord, player: Player, turn: number, tacticalBonus: number, illegalActions: number[] }[];
  isRandom: boolean;
  randomId: number;
}

export const AITraining: React.FC<Props> = ({ 
  isTraining, setIsTraining, layers, setLayers, focalRadii, setFocalRadii,
  generations, setGenerations, loss, setLoss,
  currentModelName, setCurrentModelName, trainerRef, modelRef, setIsAiLoaded
}) => {
  const [logs, setLog] = useState<string[]>(["[System] Hyper-Batch Engine Ready."]);
  const [vault, setVault] = useState<ModelMetadata[]>([]);
  const [isChampionship, setIsChampionship] = useState(false);
  const [champResults, setChampResults] = useState<{ p1: number, p2: number } | null>(null);
  const [maxTurns, setMaxTurns] = useState(250);
  const [batchSize, setBatchSize] = useState(64);
  const [parallelGames, setParallelGames] = useState(16); // High-speed parallel setting
  const [epsilon, setEpsilon] = useState(0.2);
  const [autoSaveFreq, setAutoSaveFreq] = useState(50);
  
  const [rewards, setRewards] = useState({
    p1Win: 4.0, p2Win: 5.0, p1Draw: 0.4, p2Draw: 0.6,
    line3: 0.05, line4: 0.15, line5: 0.50, block4: 0.20, block5: 0.50, 
    efficiency: -0.005, illegal: -0.05
  });

  const genRef = useRef(generations);
  useEffect(() => { setVault(getVaultMetadata()); }, []);
  useEffect(() => { genRef.current = generations; }, [generations]);

  const addLog = (msg: string) => { setLog(prev => [msg, ...prev].slice(0, 50)); };
  const parseSafeFloat = (val: string): number => {
    const parsed = parseFloat(val);
    return isNaN(parsed) ? 0 : parsed;
  };

  const initTrainer = (model: tf.LayersModel) => {
    if (!model.optimizer) {
      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError',
        metrics: ['accuracy']
      });
    }
    modelRef.current = model;
    trainerRef.current = new Trainer(model);
    setIsAiLoaded(true);
  };

  const getIOConfig = () => {
    const hexInputs = (3 * focalRadii.global * (focalRadii.global + 1) + 1) + 
                      (3 * focalRadii.self * (focalRadii.self + 1) + 1) + 
                      (3 * focalRadii.memory * (focalRadii.memory + 1) + 1) * 4;
    return { inputNodes: hexInputs + 4 + 12, outputNodes: hexInputs };
  };

  const handleCreateNew = () => {
    const { inputNodes, outputNodes } = getIOConfig();
    const model = createModel({ inputNodes, outputNodes, hiddenLayers: layers, learningRate: 0.001 });
    initTrainer(model);
    setGenerations(0); setLoss(0);
    addLog(`[System] New Dueling Model initialized.`);
  };

  const toggleTraining = async () => {
    if (!isTraining) {
      if (!modelRef.current) handleCreateNew();
      addLog("[System] Hyper-Batch Training Resumed.");
    } else {
      addLog("[System] Training paused.");
    }
    setIsTraining(!isTraining);
  };

  const performSave = async (isAuto = false) => {
    if (!trainerRef.current || !modelRef.current) return;
    try {
      const name = isAuto ? currentModelName : (prompt("Enter model name:", currentModelName) || currentModelName);
      const { inputNodes, outputNodes } = getIOConfig();
      const meta: ModelMetadata = {
        name, timestamp: Date.now(), inputNodes, outputNodes, hiddenLayers: layers,
        focalRadii: { ...focalRadii }, generation: genRef.current, maxTurns, batchSize, epsilon
      };
      await trainerRef.current.saveModel(`indexeddb://${name}`);
      const vaultData = getVaultMetadata();
      const index = vaultData.findIndex(m => m.name === name);
      if (index !== -1) vaultData[index] = meta; else vaultData.push(meta);
      localStorage.setItem('hexagon-model-vault-metadata', JSON.stringify(vaultData));
      if (!isAuto) setCurrentModelName(name);
      setVault(getVaultMetadata());
      addLog(`[System] Model saved ${isAuto ? '(Auto)' : ''}.`);
    } catch (e: any) {
      if (e.message !== "GPU_BUSY" && !isAuto) addLog("[Error] Save failed.");
    }
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
      addLog(`[System] Loaded '${name}'.`);
    } catch (e) { addLog(`[Error] Failed to load.`); }
  };

  const handleDelete = async (name: string) => {
    if (confirm(`Delete model '${name}'?`)) {
      await deleteModelFromVault(name);
      setVault(getVaultMetadata());
    }
  };

  const runChampionship = async () => {
    if (!modelRef.current || vault.length === 0) return;
    const opponentName = prompt("Enter opponent name:", vault[0].name);
    if (!opponentName) return;
    setIsChampionship(true);
    try {
      const opponentModel = await loadModelFromVault(opponentName);
      let p1Wins = 0; let p2Wins = 0;
      const config: TrainingConfig = { learningRate: 0.001, batchSize: 64, gamma: 0.95, epsilon: 0, rewards: rewards as any };
      for (let g = 0; g < 10; g++) {
        let board: BoardState = new Map(); let currentPlayer: Player = 1; let winner: Player | null = null;
        let foci: Coord[] = Array(6).fill({ q: 0, r: 0 }); let turns = 0;
        const modelIsP1 = g < 5;
        while (!winner && board.size < maxTurns * 2) {
          const m = (currentPlayer === 1 && modelIsP1) || (currentPlayer === 2 && !modelIsP1) ? modelRef.current : opponentModel;
          const result = await trainerRef.current!.playTurn(board, currentPlayer, foci, focalRadii, config, turns, maxTurns, m);
          board = result.board; winner = result.winner;
          if (result.moves.length > 0) foci[0] = result.moves[result.moves.length - 1];
          currentPlayer = currentPlayer === 1 ? 2 : 1; turns++;
        }
        if (winner) { if ((winner === 1 && modelIsP1) || (winner === 2 && !modelIsP1)) p1Wins++; else p2Wins++; }
        setChampResults({ p1: p1Wins, p2: p2Wins });
      }
    } catch (e) { addLog("[Error] Champ failed."); }
    setIsChampionship(false);
  };

  // HYPER-BATCH TRAINING LOOP
  useEffect(() => {
    let active = true;
    if (!isTraining || !trainerRef.current) return;

    const runHyperCycle = async () => {
      try {
        const currentLR = Math.max(0.0001, 0.001 * Math.pow(0.99, genRef.current / 100));
        trainerRef.current!.setLearningRate(currentLR);

        // Initialize Parallel Sessions
        const sessions: GameSession[] = Array.from({ length: parallelGames }, () => ({
          board: new Map(),
          currentPlayer: 1,
          winner: null,
          foci: Array(6).fill({ q: 0, r: 0 }),
          turns: 0,
          history: [],
          isRandom: Math.random() < 0.2,
          randomId: Math.random() > 0.5 ? 1 : 2
        }));

        let activeCount = parallelGames;

        while (activeCount > 0 && active && isTraining) {
          // 1. Gather all states for the current step (Vectorization)
          const statesToPredict: number[][] = [];
          const sessionIndices: number[] = [];

          sessions.forEach((s, i) => {
            if (!s.winner && s.turns < maxTurns) {
              const state = encodeState(s.board, s.currentPlayer, s.foci, focalRadii, s.turns, maxTurns);
              statesToPredict.push(state);
              sessionIndices.push(i);
            }
          });

          if (sessionIndices.length === 0) break;

          // 2. Batch Predict (Massive GPU efficiency)
          const batchPredictions = await trainerRef.current!.predictActionBatch(statesToPredict, epsilon);

          // 3. Process Predictions and Update Boards
          for (let i = 0; i < sessionIndices.length; i++) {
            const s = sessions[sessionIndices[i]];
            const prediction = batchPredictions[i];
            
            // Handle move (Sequential logic inside each game)
            const result = await trainerRef.current!.playTurn(s.board, s.currentPlayer, s.foci, focalRadii, { epsilon } as any, s.turns, maxTurns);
            
            // Record
            result.moves.forEach((move, moveIdx) => {
              const myMax = getMaxLine(s.board, move.q, move.r, s.currentPlayer);
              const other = (s.currentPlayer === 1 ? 2 : 1) as Player;
              const enemyMaxBefore = getMaxLine(s.board, move.q, move.r, other);
              let tBonus = 0;
              if (myMax === 3) tBonus += rewards.line3;
              if (myMax === 4) tBonus += rewards.line4;
              if (myMax === 5) tBonus += rewards.line5;
              if (enemyMaxBefore === 4) tBonus += rewards.block4;
              if (enemyMaxBefore === 5) tBonus += rewards.block5;

              s.history.push({
                boardBefore: new Map(s.board),
                foci: [...s.foci],
                move: move,
                player: s.currentPlayer,
                turn: s.turns,
                tacticalBonus: tBonus,
                illegalActions: [] // Simplified for batch speed
              });
            });

            s.board = result.board;
            s.winner = result.winner;
            if (result.moves.length > 0) s.foci[0] = result.moves[result.moves.length - 1];
            s.currentPlayer = s.currentPlayer === 1 ? 2 : 1;
            s.turns++;

            if (s.winner || s.turns >= maxTurns) activeCount--;
          }
        }

        // 4. Distribute Rewards and Save to Memory (including augmentation)
        sessions.forEach(s => {
          [1, 2].forEach(p => {
            const pExps = s.history.filter(exp => exp.player === p);
            const base = s.winner ? (s.winner === p ? (p === 1 ? rewards.p1Win : rewards.p2Win) : -1.0) : (p === 1 ? rewards.p1Draw : rewards.p2Draw);
            const totalReward = base + pExps.reduce((acc, exp) => acc + rewards.efficiency + exp.tacticalBonus, 0);
            
            pExps.forEach(exp => {
              for (let r = 0; r < 6; r++) {
                const rotBoard = rotateBoard(exp.boardBefore, r);
                const rotFoci = exp.foci.map(f => rotateCoord(f, r));
                const rotMove = rotateCoord(exp.move, r);
                const rotAction = coordToIndex(rotMove, rotFoci, focalRadii);
                if (rotAction !== -1) {
                  const rotState = encodeState(rotBoard, exp.player, rotFoci, focalRadii, exp.turn, maxTurns);
                  trainerRef.current?.addToMemory(rotState, rotAction, totalReward, [], Math.abs(totalReward) + 0.1);
                }
              }
            });
          });
        });

        // 5. Train Batch
        const l = await trainerRef.current!.trainBatch(batchSize);
        if (l) setLoss(l);

        const newGen = genRef.current + parallelGames;
        setGenerations(newGen);
        if (newGen % autoSaveFreq < parallelGames) performSave(true);
        
        if (active && isTraining) setTimeout(runHyperCycle, 10);
      } catch (err) {
        addLog("[Shield] Error detected. Restarting...");
        if (active && isTraining) setTimeout(runHyperCycle, 2000);
      }
    };

    runHyperCycle();
    return () => { active = false; };
  }, [isTraining, rewards, maxTurns, focalRadii, epsilon, batchSize, autoSaveFreq, parallelGames]);

  return (
    <div className="tab-content ai-view">
      <div className="settings-header"><h2>AI Training Lab (Hyper-Batch v2)</h2></div>
      <section className="ai-top-bar card">
        <div className="top-bar-row">
          <div className="input-group-horizontal">
            <div className="mini-input-row"><label>Max Turns</label><input type="number" value={maxTurns} onChange={e => setMaxTurns(parseSafeFloat(e.target.value))} min="10" max="500" /></div>
            <div className="mini-input-row"><label>Batch Size</label><input type="number" value={batchSize} onChange={e => setBatchSize(parseSafeFloat(e.target.value))} step={32} min="32" max="512" /></div>
            <div className="mini-input-row"><label>Parallel</label><input type="number" value={parallelGames} onChange={e => setParallelGames(Math.max(1, parseInt(e.target.value)))} step={4} min="1" max="64" /></div>
            <div className="mini-input-row"><label>Randomness</label><input type="number" value={epsilon} onChange={e => setEpsilon(parseSafeFloat(e.target.value))} step={0.05} min="0" max="1" /></div>
            <div className="mini-input-row"><label>Auto-Save</label><input type="number" value={autoSaveFreq} onChange={e => setAutoSaveFreq(Math.max(1, parseInt(e.target.value)))} min="1" max="1000" /></div>
          </div>
        </div>
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
                  <div className="vault-actions"><button onClick={() => handleLoad(m.name)}>Load</button><button className="delete-btn" onClick={() => handleDelete(m.name)}>&times;</button></div>
                </div>
              ))}
            </div>
            <button className="add-layer-btn" style={{marginTop: 'auto'}} onClick={handleCreateNew}>Initialize New Model</button>
          </section>
        </div>
        <div className="ai-side-col">
          <section className="reward-config card full-height-card">
            <h3>Reward System</h3>
            <div className="reward-grid">
              <div className="input-group"><label>Win P1/P2</label><div className="dual-input"><input type="number" value={rewards.p1Win} onChange={e => setRewards({...rewards, p1Win: parseSafeFloat(e.target.value)})} /><input type="number" value={rewards.p2Win} onChange={e => setRewards({...rewards, p2Win: parseSafeFloat(e.target.value)})} /></div></div>
              <div className="input-group"><label>Lines 3/4/5</label><div className="dual-input"><input type="number" value={rewards.line3} onChange={e => setRewards({...rewards, line3: parseSafeFloat(e.target.value)})} title="3 in a row"/><input type="number" value={rewards.line4} onChange={e => setRewards({...rewards, line4: parseSafeFloat(e.target.value)})} title="4 in a row"/><input type="number" value={rewards.line5} onChange={e => setRewards({...rewards, line5: parseSafeFloat(e.target.value)})} title="5 in a row"/></div></div>
              <div className="input-group"><label>Blocks 4/5</label><div className="dual-input"><input type="number" value={rewards.block4} onChange={e => setRewards({...rewards, block4: parseSafeFloat(e.target.value)})} title="Block enemy 4"/><input type="number" value={rewards.block5} onChange={e => setRewards({...rewards, block5: parseSafeFloat(e.target.value)})} title="Block enemy 5"/></div></div>
              <div className="input-group"><label>Illegal Move</label><input type="number" value={rewards.illegal} onChange={e => setRewards({...rewards, illegal: parseSafeFloat(e.target.value)})} /></div>
              <div className="input-group"><label>Eff.</label><input type="number" value={rewards.efficiency} onChange={e => setRewards({...rewards, efficiency: parseSafeFloat(e.target.value)})} /></div>
            </div>
          </section>
        </div>
      </div>
      <section className="training-log card" style={{marginTop: '20px'}}>
        <h3>System Logs</h3>
        <div className="log-window" style={{height: '200px'}}>{logs.map((log, i) => <p key={i}>{log}</p>)}</div>
      </section>
    </div>
  );
};
