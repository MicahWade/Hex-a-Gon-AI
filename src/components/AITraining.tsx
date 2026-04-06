import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as tf from '@tensorflow/tfjs';
import { createModel } from '../ai/modelBuilder';
import { Trainer } from '../ai/trainer';
import type { TrainingConfig } from '../ai/trainer';
import type { BoardState, Coord, Player } from '../types';
import { getVaultMetadata, loadModelFromVault, deleteModelFromVault } from '../ai/modelVault';
import type { ModelMetadata } from '../ai/modelVault';
import { encodeState, decodeMove } from '../ai/encoder';
import { getMaxLine, getTacticalMove, checkWin } from '../gameLogic';
import { coordToString } from '../types';

// VITE WORKER IMPORT (More stable than new URL constructor)
import AugmentationWorker from '../ai/augmentationWorker?worker';

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
  maxTurns: number;
  setMaxTurns: (val: number) => void;
  batchSize: number;
  setBatchSize: (val: number) => void;
  epsilon: number;
  setEpsilon: (val: number) => void;
  parallelGames: number;
  setParallelGames: (val: number) => void;
}

const EPSILON_BUCKETS = [0.05, 0.15, 0.35, 0.65];
const BUCKET_LABELS = ['Conservative', 'Standard', 'Aggressive', 'Chaos'];

export const AITraining: React.FC<Props> = ({ 
  isTraining, setIsTraining, layers, setLayers, focalRadii, setFocalRadii,
  generations, setGenerations, loss, setLoss,
  currentModelName, setCurrentModelName, trainerRef, modelRef, setIsAiLoaded,
  maxTurns, setMaxTurns, batchSize, setBatchSize, epsilon, setEpsilon, parallelGames, setParallelGames
}) => {
  const [logs, setLog] = useState<string[]>(["[System] Safety-shield active."]);
  const [vault, setVault] = useState<ModelMetadata[]>([]);
  const [isChampionship, setIsChampionship] = useState(false);
  const [champResults, setChampResults] = useState<{ p1: number, p2: number } | null>(null);
  const [autoSaveFreq, setAutoSaveFreq] = useState(50);
  const [isStopping, setIsStopping] = useState(false);
  const [saveNameInput, setSaveNameInput] = useState(currentModelName);
  
  const [rewards, setRewards] = useState({
    p1Win: 4.0, p2Win: 5.0, p1Draw: 0.4, p2Draw: 0.6,
    line3: 0.05, line4: 0.15, line5: 0.50, block4: 0.20, block5: 0.50, 
    efficiency: -0.005, illegal: -0.05
  });

  const bucketStatsRef = useRef<number[]>([1, 1, 1, 1]);
  const [visibleBucketStats, setVisibleBucketStats] = useState<number[]>([1, 1, 1, 1]);
  const bucketHistory = useRef<{ bucketIdx: number, win: boolean }[]>([]);

  const genRef = useRef(generations);
  const lastUiUpdate = useRef(0);
  const pendingGen = useRef(generations);
  const pendingLoss = useRef(loss);
  const workerRef = useRef<Worker | null>(null);
  const workerCallbacks = useRef<Map<string, (data: any) => void>>(new Map());

  useEffect(() => {
    // Initialize Worker using Vite class
    const worker = new AugmentationWorker();
    worker.onmessage = (e) => {
      const { requestId, data } = e.data;
      const callback = workerCallbacks.current.get(requestId);
      if (callback) { callback(data); workerCallbacks.current.delete(requestId); }
    };
    workerRef.current = worker;
    return () => { worker.terminate(); };
  }, []);

  useEffect(() => { setVault(getVaultMetadata()); }, []);
  useEffect(() => { 
    genRef.current = generations;
    pendingGen.current = generations;
  }, [generations]);

  useEffect(() => { setSaveNameInput(currentModelName); }, [currentModelName]);

  const addLog = (msg: string) => { setLog(prev => [msg, ...prev].slice(0, 30)); };
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
    addLog(`[System] New model initialized.`);
  };

  const toggleTraining = async () => {
    if (!isTraining) {
      if (!modelRef.current) handleCreateNew();
      addLog("[System] Training resumed.");
      setIsTraining(true);
    } else {
      addLog("[System] Training stopping...");
      setIsStopping(true);
      setIsTraining(false);
    }
  };

  const performSave = async (isAuto = false) => {
    if (!trainerRef.current || !modelRef.current) return;
    try {
      const name = isAuto ? currentModelName : (saveNameInput || currentModelName);
      const { inputNodes, outputNodes } = getIOConfig();
      const meta: ModelMetadata = {
        name, timestamp: Date.now(), inputNodes, outputNodes, hiddenLayers: layers,
        focalRadii: { ...focalRadii }, generation: genRef.current, maxTurns, batchSize, epsilon,
        parallelGames: parallelGames
      };
      await trainerRef.current.saveModel(`indexeddb://${name}`);
      const vaultData = getVaultMetadata();
      const index = vaultData.findIndex(m => m.name === name);
      if (index !== -1) vaultData[index] = meta; else vaultData.push(meta);
      localStorage.setItem('hexagon-model-vault-metadata', JSON.stringify(vaultData));
      localStorage.setItem('hexagon-last-model-name', name);
      if (!isAuto) setCurrentModelName(name);
      setVault(getVaultMetadata());
      if (!isAuto) addLog(`[System] Saved '${name}'.`);
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
        if (meta.parallelGames !== undefined) setParallelGames(meta.parallelGames);
      }
      initTrainer(model);
      if (trainerRef.current) trainerRef.current.clearMemory();
      setCurrentModelName(name);
      addLog(`[System] Loaded '${name}'.`);
      localStorage.setItem('hexagon-last-model-name', name);
    } catch (e) { addLog(`[Error] Failed to load.`); }
  };

  const handleImportPython = async () => {
    addLog("[System] Initializing Raw Weight Sync...");
    try {
      // 1. Fetch the manifest
      const manifestRes = await fetch('/python_model/manifest.json');
      if (!manifestRes.ok) throw new Error("Manifest not found");
      const manifest = await manifestRes.json();

      // 2. Build the Model Architecture manually in TFJS
      const { inputNodes, outputNodes } = getIOConfig();
      const model = tf.sequential();
      
      // Feature Layers (Standardized 7x3500)
      model.add(tf.layers.dense({ units: 3500, activation: 'relu', inputShape: [inputNodes], name: 'dense_0' }));
      for (let i = 1; i < 7; i++) {
        model.add(tf.layers.dense({ units: 3500, activation: 'relu', name: `dense_${i}` }));
      }
      // Dueling Head (Simplified combined output for loading)
      model.add(tf.layers.dense({ units: outputNodes, activation: 'linear', name: 'output_head' }));

      // 3. Inject Weights Layer-by-Layer
      addLog("[System] Injecting weights into neurons...");
      
      const loadLayer = async (layerName: string, weightFile: string, biasFile: string, weightShape: number[], biasShape: number[]) => {
        const [wRes, bRes] = await Promise.all([
          fetch(`/python_model/${weightFile}`),
          fetch(`/python_model/${biasFile}`)
        ]);
        const [wBuf, bBuf] = await Promise.all([wRes.arrayBuffer(), bRes.arrayBuffer()]);
        
        const layer = model.getLayer(layerName);
        layer.setWeights([
          tf.tensor(new Float32Array(wBuf), weightShape),
          tf.tensor(new Float32Array(bBuf), biasShape)
        ]);
      };

      // Load Feature Weights
      for (let i = 0; i < 7; i++) {
        const wInfo = manifest.find((m: any) => m.name === `feature_${i}_w`);
        const bInfo = manifest.find((m: any) => m.name === `feature_${i}_b`);
        await loadLayer(`dense_${i}`, wInfo.file, bInfo.file, wInfo.shape, bInfo.shape);
      }

      // Load Output Head (Combination of Advantage + Value from Python)
      // Note: In this raw injection, we treat the final layer as a single dense layer
      const awInfo = manifest.find((m: any) => m.name === `advantage_w`);
      const abInfo = manifest.find((m: any) => m.name === `advantage_b`);
      await loadLayer('output_head', awInfo.file, abInfo.file, awInfo.shape, abInfo.shape);

      // 4. Save to Vault
      const name = "python-raw-brain";
      const meta: ModelMetadata = {
        name, timestamp: Date.now(), inputNodes, outputNodes, 
        hiddenLayers: [3500, 3500, 3500, 3500, 3500, 3500, 3500],
        focalRadii: { ...focalRadii }, generation: 0, maxTurns, batchSize, epsilon,
        parallelGames: 1
      };
      
      await model.save(`indexeddb://${name}`);
      const vaultData = getVaultMetadata();
      const index = vaultData.findIndex(m => m.name === name);
      if (index !== -1) vaultData[index] = meta; else vaultData.push(meta);
      localStorage.setItem('hexagon-model-vault-metadata', JSON.stringify(vaultData));
      
      setVault(getVaultMetadata());
      addLog("[System] GPU Brain injected successfully!");
    } catch (e) {
      console.error(e);
      addLog("[Error] Sync failed. Run 'python export_raw.py' first.");
    }
  };

  const handleExportMemory = () => {
    if (!trainerRef.current) return;
    const memory = trainerRef.current.getMemory();
    const blob = new Blob([JSON.stringify(memory)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `memory_${Date.now()}.json`;
    a.click();
    addLog(`[System] Exported ${memory.length} experiences.`);
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

  useEffect(() => {
    let active = true;
    if (!isTraining || !trainerRef.current) return;

    const runSingleGame = async () => {
      const currentLR = Math.max(0.0001, 0.001 * Math.pow(0.99, genRef.current / 100));
      trainerRef.current!.setLearningRate(currentLR);
      
      const stats = bucketStatsRef.current;
      const rawTotal = stats.reduce((a, b) => a + b, 0);
      
      // Calculate weights with a 10% minimum floor
      const minWeight = rawTotal * 0.10; 
      const flooredStats = stats.map(s => Math.max(s, minWeight));
      const totalWeight = flooredStats.reduce((a, b) => a + b, 0);

      let r = Math.random() * totalWeight;
      let bucketIdx = 0;
      for (let i = 0; i < flooredStats.length; i++) {
        r -= flooredStats[i];
        if (r <= 0) { bucketIdx = i; break; }
      }
      const chosenEpsilon = EPSILON_BUCKETS[bucketIdx];

      const config: TrainingConfig = { learningRate: currentLR, batchSize, gamma: 0.95, epsilon: chosenEpsilon, rewards: rewards as any };
      
      let board: BoardState = new Map();
      let currentPlayer: Player = 1;
      let winner: Player | null = null;
      let foci: Coord[] = Array(6).fill({ q: 0, r: 0 });
      let turns = 0;
      const gameHistory: { boardBefore: BoardState, foci: Coord[], move: Coord, player: Player, turn: number, tacticalBonus: number, illegalActions: number[] }[] = [];

      const randVal = Math.random();
      const botType: 'NONE' | 'RANDOM' | 'TACTICAL' = randVal < 0.15 ? 'RANDOM' : (randVal < 0.30 ? 'TACTICAL' : 'NONE');
      const botPlayerId = botType !== 'NONE' ? (Math.random() > 0.5 ? 1 : 2) : 0;

      while (!winner && turns < maxTurns && active && isTraining) {
        if (turns % 20 === 0) await tf.nextFrame();
        const boardBefore = new Map(board);
        const currentFoci = [...foci];
        let result: { board: BoardState; moves: Coord[]; winner: Player | null; actionIndices: number[] };

        if (currentPlayer === botPlayerId) {
          const moves: Coord[] = [];
          let currentBoard = new Map(board);
          const moveCount = (turns === 0 && currentPlayer === 1) ? 1 : 2;
          for (let m = 0; m < moveCount; m++) {
            let move: Coord;
            if (botType === 'RANDOM') {
              const allMoves = await trainerRef.current!.getTopMoves(encodeState(currentBoard, currentPlayer, foci, focalRadii, turns, maxTurns), 50);
              move = decodeMove(allMoves[Math.floor(Math.random() * allMoves.length)].idx, foci, focalRadii);
            } else {
              move = getTacticalMove(currentBoard, currentPlayer, 0.5);
            }
            const key = coordToString(move);
            if (!currentBoard.has(key)) { currentBoard.set(key, currentPlayer); moves.push(move); if (checkWin(currentBoard, move.q, move.r, currentPlayer)) { winner = currentPlayer; break; } }
          }
          result = { board: currentBoard, moves, winner, actionIndices: [] };
        } else {
          result = await trainerRef.current!.playTurn(board, currentPlayer, foci, focalRadii, { ...config, epsilon: chosenEpsilon }, turns, maxTurns);
        }
        
        const illegalActions: number[] = [];
        if (currentPlayer !== botPlayerId && board.size > 0) {
          const topOptions = await trainerRef.current!.getTopMoves(encodeState(board, currentPlayer, foci, focalRadii, turns, maxTurns), 3);
          topOptions.forEach(opt => { if (board.has(coordToString(decodeMove(opt.idx, foci, focalRadii)))) illegalActions.push(opt.idx); });
        }

        result.moves.forEach((move) => {
          const myMax = getMaxLine(board, move.q, move.r, currentPlayer);
          const other = (currentPlayer === 1 ? 2 : 1) as Player;
          const enemyMaxBefore = getMaxLine(board, move.q, move.r, other);
          let tBonus = 0;
          if (myMax === 3) tBonus += rewards.line3; if (myMax === 4) tBonus += rewards.line4; if (myMax === 5) tBonus += rewards.line5;
          if (enemyMaxBefore === 4) tBonus += rewards.block4; if (enemyMaxBefore === 5) tBonus += rewards.block5;
          gameHistory.push({ boardBefore: boardBefore, foci: currentFoci, move: move, player: currentPlayer, turn: turns, tacticalBonus: tBonus, illegalActions: illegalActions });
        });

        board = result.board; winner = result.winner;
        if (result.moves.length > 0) foci[0] = result.moves[result.moves.length - 1];
        currentPlayer = currentPlayer === 1 ? 2 : 1; turns++;
      }

      const playerResults = [1, 2].map(p => {
        const pExps = gameHistory.filter(exp => exp.player === p);
        const base = winner ? (winner === p ? (p === 1 ? rewards.p1Win : rewards.p2Win) : -1.0) : (p === 1 ? rewards.p1Draw : rewards.p2Draw);
        const totalReward = base + pExps.reduce((acc, exp) => acc + rewards.efficiency + exp.tacticalBonus, 0);
        return { player: p, experiences: pExps, total: totalReward };
      });

      if (winner) {
        const winIdx = playerResults.findIndex(r => r.player === winner);
        const loseIdx = playerResults.findIndex(r => r.player !== winner);
        if (playerResults[winIdx].total <= playerResults[loseIdx].total) playerResults[winIdx].total = playerResults[loseIdx].total + 0.1;
        
        const isBotWinner = winner === botPlayerId;
        bucketHistory.current.push({ bucketIdx, win: !isBotWinner });
        if (bucketHistory.current.length > 1000) bucketHistory.current.shift();
        
        const newStats = [1, 1, 1, 1];
        bucketHistory.current.forEach(h => { if (h.win) newStats[h.bucketIdx]++; });
        bucketStatsRef.current = newStats;
      }

      if (workerRef.current) {
        const requestId = Math.random().toString(36).substring(7);
        const processedBatch = await new Promise<any[]>((resolve) => {
          workerCallbacks.current.set(requestId, resolve);
          workerRef.current!.postMessage({ requestId, experiences: playerResults, focalRadii, maxTurns, rewards });
        });
        processedBatch.forEach(item => { trainerRef.current?.addToMemory(item.state, item.action, item.reward, null, item.priority); });
      }

      return { winner, turns, botType, bucketIdx };
    };

    const runCycle = async () => {
      try {
        if (!active || !isTraining) return;
        const games = Array.from({ length: parallelGames }, () => runSingleGame());
        const results = await Promise.all(games);
        const l = await trainerRef.current!.trainBatch(batchSize);
        
        results.forEach(res => {
          if (res.winner) {
            const opponentStr = res.botType === 'NONE' ? 'Self' : (res.botType === 'RANDOM' ? 'Random' : 'Tactical');
            addLog(`[Game] P${res.winner} won (${opponentStr}) • Eps: ${EPSILON_BUCKETS[res.bucketIdx]}`);
          }
        });

        pendingGen.current += results.length;
        if (l) pendingLoss.current = l;
        const now = Date.now();
        if (now - lastUiUpdate.current > 500) { 
          setGenerations(pendingGen.current); 
          setLoss(pendingLoss.current); 
          setVisibleBucketStats([...bucketStatsRef.current]);
          lastUiUpdate.current = now; 
        }
        if (pendingGen.current % autoSaveFreq < results.length) performSave(true);
        
        if (active && isTraining) { 
          await tf.nextFrame();
          setTimeout(runCycle, 50); 
        } else { setIsStopping(false); addLog("[System] Training paused."); }
      } catch (err) { if (active && isTraining) setTimeout(runCycle, 2000); else setIsStopping(false); }
    };

    runCycle();
    return () => { active = false; };
  }, [isTraining, rewards, maxTurns, focalRadii, epsilon, batchSize, autoSaveFreq, parallelGames]);

  const vaultPanel = useMemo(() => (
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
      <div style={{display: 'flex', gap: '10px', marginTop: 'auto', flexWrap: 'wrap'}}>
        <button className="add-layer-btn" style={{flex: '1 1 45%'}} onClick={handleCreateNew}>Initialize New Model</button>
        <button className="add-layer-btn" style={{flex: '1 1 45%', backgroundColor: '#9b59b6'}} onClick={handleImportPython}>Sync Python Model</button>
        <button className="add-layer-btn" style={{flex: '1 1 100%', backgroundColor: '#27ae60'}} onClick={handleExportMemory}>Export Training Data (.json)</button>
      </div>
    </section>
  ), [vault, currentModelName]);

  return (
    <div className="tab-content ai-view">
      <div className="settings-header"><h2>AI Training Lab</h2></div>
      <section className="ai-top-bar card">
        <div className="top-bar-row">
          <div className="input-group-horizontal">
            <div className="mini-input-row"><label>Max Turns</label><input type="number" value={maxTurns} onChange={e => setMaxTurns(parseSafeFloat(e.target.value))} min="10" max="500" /></div>
            <div className="mini-input-row"><label>Batch Size</label><input type="number" value={batchSize} onChange={e => setBatchSize(parseSafeFloat(e.target.value))} min="32" max="1024" /></div>
            <div className="mini-input-row"><label>Parallel</label><input type="number" value={parallelGames} onChange={e => setParallelGames(Math.max(1, parseInt(e.target.value)))} min="1" max="64" /></div>
            <div className="mini-input-row"><label>Auto-Save</label><input type="number" value={autoSaveFreq} onChange={e => setAutoSaveFreq(Math.max(1, parseInt(e.target.value)))} min="1" max="1000" /></div>
          </div>
        </div>
        <div className="top-bar-row" style={{marginTop: '15px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '15px'}}>
          <div className="action-buttons" style={{flex: 1, display: 'flex', alignItems: 'center', gap: '10px'}}>
            <button className={isStopping ? 'stopping-btn' : (isTraining ? 'stop-btn' : 'start-btn')} onClick={toggleTraining} disabled={isStopping} style={{flex: '0 0 auto', width: '140px'}}>
              {isStopping ? 'Stopping...' : (isTraining ? 'Stop' : 'Start Training')}
            </button>
            <div className="save-container" style={{display: 'flex', gap: '5px', flex: 1, maxWidth: '300px'}}>
              <input type="text" value={saveNameInput} onChange={e => setSaveNameInput(e.target.value)} placeholder="Model Name" className="save-input" style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', color: 'white', padding: '8px 12px', borderRadius: '6px', fontSize: '13px', flex: 1 }} />
              <button className="reset-btn" onClick={() => performSave(false)} disabled={isTraining || isStopping} style={{width: 'auto', padding: '0 15px'}}>Save</button>
            </div>
            <button className="recommend-btn" onClick={runChampionship} disabled={isTraining || isChampionship} style={{marginLeft: 'auto'}}>Champ</button>
          </div>
          <div className="stats-section" style={{marginLeft: '20px'}}>
            <div className="stat-pill"><span>Gen</span> <strong>{generations}</strong></div>
            <div className="stat-pill"><span>Loss</span> <strong>{loss.toFixed(4)}</strong></div>
            {champResults && <div className="champ-pill"><span>Score</span> <strong>{champResults.p1}-{champResults.p2}</strong></div>}
          </div>
        </div>
      </section>
      <div className="ai-grid">
        <div className="ai-main-col">{vaultPanel}</div>
        <div className="ai-side-col">
          <section className="reward-config card full-height-card">
            <h3>Exploration Pulse</h3>
            <div className="epsilon-pulse-grid" style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '20px'}}>
              {EPSILON_BUCKETS.map((val, i) => {
                const rawTotal = visibleBucketStats.reduce((a,b)=>a+b,0);
                const minWeight = rawTotal * 0.10; // 10% UI Floor
                const flooredStats = visibleBucketStats.map(s => Math.max(s, minWeight));
                const totalWithFloor = flooredStats.reduce((a,b)=>a+b,0);
                const percentage = Math.round((flooredStats[i] / totalWithFloor) * 100);
                
                return (
                  <div key={val} className="pulse-item" style={{background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)'}}>
                    <div style={{fontSize: '10px', color: 'rgba(255,255,255,0.5)'}}>{BUCKET_LABELS[i]}</div>
                    <div style={{fontSize: '14px', fontWeight: 'bold', display: 'flex', justifyContent: 'space-between'}}>
                      <span>{val}</span>
                      <span style={{color: '#2ecc71'}}>{percentage}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
            <h3>Reward System</h3>
            <div className="reward-grid">
              <div className="input-group"><label>Win P1/P2</label><div className="dual-input"><input type="number" value={rewards.p1Win} onChange={e => setRewards({...rewards, p1Win: parseSafeFloat(e.target.value)})} /><input type="number" value={rewards.p2Win} onChange={e => setRewards({...rewards, p2Win: parseSafeFloat(e.target.value)})} /></div></div>
              <div className="input-group"><label>Lines 3/4/5</label><div className="dual-input"><input type="number" value={rewards.line3} onChange={e => setRewards({...rewards, line3: parseSafeFloat(e.target.value)})} /><input type="number" value={rewards.line4} onChange={e => setRewards({...rewards, line4: parseSafeFloat(e.target.value)})} /><input type="number" value={rewards.line5} onChange={e => setRewards({...rewards, line5: parseSafeFloat(e.target.value)})} /></div></div>
              <div className="input-group"><label>Blocks 4/5</label><div className="dual-input"><input type="number" value={rewards.block4} onChange={e => setRewards({...rewards, block4: parseSafeFloat(e.target.value)})} /><input type="number" value={rewards.block5} onChange={e => setRewards({...rewards, block5: parseSafeFloat(e.target.value)})} /></div></div>
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
