import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { createModel } from '../ai/modelBuilder';
import { Trainer } from '../ai/trainer';
import type { TrainingConfig } from '../ai/trainer';
import type { BoardState, Coord, Player } from '../types';

interface Props {
  isTraining: boolean;
  setIsTraining: (val: boolean) => void;
  layers: number[];
}

export const AITraining: React.FC<Props> = ({ isTraining, setIsTraining, layers }) => {
  const [generations, setGenerations] = useState(0);
  const [loss, setLoss] = useState<number>(0);
  const [logs, setLog] = useState<string[]>(["[System] Ready for training."]);
  
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

  // Auto-load model on mount
  useEffect(() => {
    const autoLoad = async () => {
      try {
        const savedModel = await tf.loadLayersModel('indexeddb://hex-a-gon-model');
        modelRef.current = savedModel;
        trainerRef.current = new Trainer(savedModel);
        addLog("[System] Previous model loaded from storage.");
      } catch (e) {
        addLog("[System] No saved model found.");
      }
    };
    autoLoad();
  }, []);

  const saveModel = async () => {
    if (modelRef.current) {
      await modelRef.current.save('indexeddb://hex-a-gon-model');
      addLog("[System] Model saved.");
    }
  };

  const addLog = (msg: string) => {
    setLog(prev => [msg, ...prev].slice(0, 50));
  };

  const toggleTraining = async () => {
    if (!isTraining) {
      addLog("[System] Initializing model...");
      
      // Calculate node counts based on current architecture
      // For simplicity, we use the default radii from our research phase
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

      modelRef.current = model;
      trainerRef.current = new Trainer(model);
      addLog("[System] Model compiled. Starting self-play...");
    } else {
      addLog("[System] Training paused.");
    }
    setIsTraining(!isTraining);
  };

  // The Background Training Loop
  useEffect(() => {
    let active = true;
    if (!isTraining || !trainerRef.current) return;

    const runTrainingCycle = async () => {
      const config: TrainingConfig = {
        learningRate: 0.001,
        batchSize: 32,
        gamma: 0.95,
        epsilon: 0.2, // 20% random moves for exploration
        rewards
      };

      const radii = { global: 14, self: 8, memory: 6 };
      let board: BoardState = new Map();
      let currentPlayer: Player = 1;
      let turnCount = 0;
      let winner: Player | null = null;

      // Simple starting focus (center)
      let foci: Coord[] = Array(6).fill({ q: 0, r: 0 });

      while (!winner && turnCount < 100 && active && isTraining) {
        const result = await trainerRef.current!.selfPlayTurn(board, currentPlayer, foci, radii, config);
        board = result.board;
        winner = result.winner;
        
        // Update focus to latest move
        if (result.moves.length > 0) {
          const latest = result.moves[result.moves.length - 1];
          foci[0] = latest; // Global focus
          // ... Shift other foci
        }

        currentPlayer = currentPlayer === 1 ? 2 : 1;
        turnCount++;

        // Train on a batch periodically
        if (turnCount % 10 === 0) {
          const currentLoss = await trainerRef.current!.trainBatch(32);
          if (currentLoss) setLoss(currentLoss);
        }
      }

      if (winner) addLog(`[Game] Player ${winner} won in ${turnCount} turns.`);
      setGenerations(g => g + 1);
      
      if (active && isTraining) {
        setTimeout(runTrainingCycle, 100);
      }
    };

    runTrainingCycle();
    return () => { active = false; };
  }, [isTraining, rewards]);

  return (
    <div className="tab-content ai-view">
      <div className="settings-header">
        <h2>AI Training Lab</h2>
        <p className="section-desc">Training control center and incentive configuration.</p>
      </div>
      
      <div className="ai-grid">
        <div className="ai-main-col">
          <section className="training-log card">
            <h3>Training Log</h3>
            <div className="log-window">
              {logs.map((log, i) => <p key={i}>{log}</p>)}
            </div>
          </section>

          <section className="reward-config card">
            <h3>Reward System</h3>
            <p className="section-desc">Adjust training incentives.</p>
            <div className="reward-grid">
              <div className="input-group">
                <label>P1 Win Weight</label>
                <input type="number" value={rewards.p1Win} step={0.1} onChange={e => setRewards({...rewards, p1Win: parseFloat(e.target.value)})} />
              </div>
              <div className="input-group">
                <label>P2 Win Weight</label>
                <input type="number" value={rewards.p2Win} step={0.1} onChange={e => setRewards({...rewards, p2Win: parseFloat(e.target.value)})} />
              </div>
              <div className="input-group">
                <label>P1 Draw</label>
                <input type="number" value={rewards.p1Draw} step={0.1} onChange={e => setRewards({...rewards, p1Draw: parseFloat(e.target.value)})} />
              </div>
              <div className="input-group">
                <label>P2 Draw</label>
                <input type="number" value={rewards.p2Draw} step={0.1} onChange={e => setRewards({...rewards, p2Draw: parseFloat(e.target.value)})} />
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
              <button className={isTraining ? 'stop-btn' : 'start-btn'} onClick={toggleTraining}>
                {isTraining ? 'Stop Training' : 'Start Training'}
              </button>
              <button className="reset-btn" onClick={saveModel} style={{ border: '1px solid var(--accent-color)', color: 'var(--accent-color)' }}>
                Save Progress
              </button>
            </div>

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
