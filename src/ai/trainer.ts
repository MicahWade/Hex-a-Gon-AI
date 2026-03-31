import * as tf from '@tensorflow/tfjs';
import type { BoardState, Player, Coord } from '../types';
import { coordToString } from '../types';
import { checkWin } from '../gameLogic';
import { encodeState, decodeMove } from './encoder';

export interface TrainingConfig {
  learningRate: number;
  batchSize: number;
  gamma: number; 
  epsilon: number; 
  rewards: {
    p1Win: number;
    p2Win: number;
    p1Draw: number;
    p2Draw: number;
    line3: number;
    line4: number;
    line5: number;
    block4: number;
    block5: number;
    efficiency: number;
    illegal: number;
  };
}

export class Trainer {
  private model: tf.LayersModel;
  private memory: { state: number[]; action: number; reward: number; nextState: number[] | null; priority: number }[] = [];
  private maxMemory = 2000;
  private isFitting = false; 

  constructor(model: tf.LayersModel) {
    this.model = model;
  }

  setLearningRate(lr: number) {
    if (this.model.optimizer) {
      (this.model.optimizer as any).learningRate = lr;
    }
  }

  async getTopMoves(state: number[], count: number = 3): Promise<{prob: number, idx: number}[]> {
    const inputSize = this.model.inputs[0].shape[1] as number;
    return tf.tidy(() => {
      const input = tf.tensor2d(new Float32Array(state), [1, inputSize]);
      const probs = (this.model.predict(input) as tf.Tensor).dataSync();
      return Array.from(probs)
        .map((prob, idx) => ({ prob, idx }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, count);
    });
  }

  async playTurn(
    board: BoardState,
    player: Player,
    foci: Coord[],
    radii: any,
    config: TrainingConfig,
    turn: number,
    maxTurns: number,
    specificModel?: tf.LayersModel
  ): Promise<{ board: BoardState; moves: Coord[]; winner: Player | null; actionIndices: number[] }> {
    const moves: Coord[] = [];
    const actionIndices: number[] = [];
    let currentBoard = new Map(board);
    let winner: Player | null = null;
    const modelToUse = specificModel || this.model;

    const isFirstMove = board.size === 0;
    const moveCount = isFirstMove ? 1 : 2;
    
    for (let i = 0; i < moveCount; i++) {
      const state = encodeState(currentBoard, player, foci, radii, turn, maxTurns);
      let action: number = 0;
      let move: Coord = foci[0];

      if (isFirstMove) {
        move = { q: 0, r: 0 };
        action = 0;
      } else {
        const prediction = await this.predictActionSingle(state, config.epsilon, modelToUse);
        
        if (typeof prediction === 'number') {
          action = prediction;
          move = decodeMove(action, foci, radii);
        } else {
          let foundValid = false;
          for (const option of prediction) {
            const candidate = decodeMove(option.idx, foci, radii);
            if (!currentBoard.has(coordToString(candidate))) {
              action = option.idx;
              move = candidate;
              foundValid = true;
              break;
            }
          }
          if (!foundValid) { action = 0; move = foci[0]; }
        }
      }

      actionIndices.push(action);
      const key = coordToString(move);
      if (!currentBoard.has(key)) {
        currentBoard.set(key, player);
        moves.push(move);
        if (checkWin(currentBoard, move.q, move.r, player)) {
          winner = player;
          break;
        }
      }
    }

    return { board: currentBoard, moves, winner, actionIndices };
  }

  private async predictActionSingle(state: number[], epsilon: number, model: tf.LayersModel): Promise<{prob: number, idx: number}[] | number> {
    const inputSize = model.inputs[0].shape[1] as number;
    const outputSize = model.outputs[0].shape[1] as number;
    if (Math.random() < epsilon) return Math.floor(Math.random() * outputSize);

    return tf.tidy(() => {
      const input = tf.tensor2d(new Float32Array(state), [1, inputSize]);
      const prediction = model.predict(input) as tf.Tensor;
      const probs = prediction.dataSync();
      return Array.from(probs).map((prob, idx) => ({ prob, idx })).sort((a, b) => b.prob - a.prob);
    });
  }

  async trainBatch(batchSize: number) {
    if (this.memory.length < batchSize || this.isFitting) return null;

    const inputSize = this.model.inputs[0].shape[1] as number;
    const outputSize = this.model.outputs[0].shape[1] as number;

    const validMemory = this.memory.filter(m => m.state.length === inputSize);
    if (validMemory.length < batchSize) return null;

    this.isFitting = true;
    
    // Prioritized sampling
    const totalPriority = validMemory.reduce((sum, m) => sum + m.priority, 0);
    const batch: typeof validMemory = [];
    for (let i = 0; i < batchSize; i++) {
      let rand = Math.random() * totalPriority;
      for (const m of validMemory) {
        rand -= m.priority;
        if (rand <= 0) { batch.push(m); break; }
      }
    }

    const statesData = new Float32Array(batchSize * inputSize);
    for (let i = 0; i < batchSize; i++) {
      statesData.set(batch[i].state, i * inputSize);
    }

    const states = tf.tensor2d(statesData, [batchSize, inputSize]);
    const targets = tf.tidy(() => {
      const currentPredictions = this.model.predict(states) as tf.Tensor;
      const targetData = currentPredictions.arraySync() as number[][];
      batch.forEach((m, i) => {
        if (m.action < outputSize) targetData[i][m.action] = m.reward;
      });
      return tf.tensor2d(targetData);
    });

    try {
      const result = await this.model.fit(states, targets, { 
        epochs: 1, 
        batchSize: batchSize,
        verbose: 0 
      });
      return result.history.loss[0] as number;
    } finally {
      states.dispose();
      targets.dispose();
      this.isFitting = false;
    }
  }

  async saveModel(url: string) {
    if (this.isFitting) throw new Error("GPU_BUSY");
    this.isFitting = true;
    try { await this.model.save(url); } finally { this.isFitting = false; }
  }

  clearMemory() { this.memory = []; }

  addToMemory(state: number[], action: number, reward: number, nextState: number[] | null, priority: number = 1.0) {
    const expectedSize = (this.model.inputs[0].shape[1] as number);
    if (state.length !== expectedSize) return;
    this.memory.push({ state, action, reward, nextState, priority });
    if (this.memory.length > this.maxMemory) this.memory.shift();
  }

  public isBusy(): boolean { return this.isFitting; }
}
