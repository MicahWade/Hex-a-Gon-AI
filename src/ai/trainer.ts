import * as tf from '@tensorflow/tfjs';
import type { BoardState, Player, Coord } from '../types';
import { coordToString } from '../types';
import { checkWin } from '../gameLogic';
import { encodeState, decodeMove } from './encoder';

export interface TrainingConfig {
  learningRate: number;
  batchSize: number;
  gamma: number; // Discount factor
  epsilon: number; // Exploration rate
  rewards: {
    p1Win: number;
    p2Win: number;
    p1Draw: number;
    p2Draw: number;
    threat: number;
    efficiency: number;
  };
}

/**
 * Handles the background self-play and model optimization.
 */
export class Trainer {
  private model: tf.LayersModel;
  private memory: { state: number[]; action: number; reward: number; nextState: number[] | null }[] = [];
  private maxMemory = 2000; // Reduced for Firefox/Zen stability

  constructor(model: tf.LayersModel) {
    this.model = model;
  }

  /**
   * Performs one turn of play (2 moves) using a specific model.
   */
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

    // AI makes sequential moves
    const isFirstMove = board.size === 0;
    const moveCount = isFirstMove ? 1 : 2;
    
    for (let i = 0; i < moveCount; i++) {
      const state = encodeState(currentBoard, player, foci, radii, turn, maxTurns);
      let action: number;
      let move: Coord;

      if (isFirstMove) {
        move = { q: 0, r: 0 };
        action = 0; 
      } else {
        const prediction = await this.predictAction(state, config.epsilon, modelToUse);
        
        if (typeof prediction === 'number') {
          // Random epsilon move
          action = prediction;
          move = decodeMove(action, foci, radii);
        } else {
          // Search for first valid move in sorted list
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
          if (!foundValid) {
            action = 0;
            move = foci[0];
          }
        }
      }

      actionIndices.push(action);
      const key = coordToString(move!);
      if (!currentBoard.has(key)) {
        currentBoard.set(key, player);
        moves.push(move!);

        if (checkWin(currentBoard, move!.q, move!.r, player)) {
          winner = player;
          break;
        }
      }
    }

    return { board: currentBoard, moves, winner, actionIndices };
  }

  private async predictAction(state: number[], epsilon: number, modelOverride?: tf.LayersModel): Promise<{prob: number, idx: number}[] | number> {
    const model = modelOverride || this.model;
    const inputSize = model.inputs[0].shape[1] as number;
    const outputSize = model.outputs[0].shape[1] as number;
    
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * outputSize);
    }

    return tf.tidy(() => {
      // Adapt state to model input size (supports backward compatibility)
      let adaptedState = state;
      if (state.length > inputSize) {
        adaptedState = state.slice(0, inputSize);
      } else if (state.length < inputSize) {
        adaptedState = [...state, ...new Array(inputSize - state.length).fill(0)];
      }

      const inputData = new Float32Array(adaptedState);
      const input = tf.tensor2d(inputData, [1, inputSize]);
      const prediction = model.predict(input) as tf.Tensor;
      
      const probs = prediction.dataSync();
      return Array.from(probs)
        .map((prob, idx) => ({ prob, idx }))
        .sort((a, b) => b.prob - a.prob);
    });
  }

  /**
   * Trains the model on a random batch of experiences from memory.
   */
  async trainBatch(batchSize: number) {
    if (this.memory.length < batchSize) return null;

    const inputSize = this.model.inputs[0].shape[1] as number;
    const outputSize = this.model.outputs[0].shape[1] as number;

    // Filter memory to ensure only states matching current model input size are used
    const validMemory = this.memory.filter(m => m.state.length === inputSize);
    if (validMemory.length < batchSize) return null;

    const batch = validMemory.sort(() => 0.5 - Math.random()).slice(0, batchSize);
    
    // Create flat Float32Arrays to avoid JS nested array overhead and RangeErrors
    const statesData = new Float32Array(batchSize * inputSize);
    for (let i = 0; i < batchSize; i++) {
      statesData.set(batch[i].state, i * inputSize);
    }

    const states = tf.tensor2d(statesData, [batchSize, inputSize]);
    
    const targets = tf.tidy(() => {
      const currentPredictions = this.model.predict(states) as tf.Tensor;
      // We must use arraySync here to modify, but we do it on the whole batch at once
      const targetData = currentPredictions.arraySync() as number[][];

      batch.forEach((m, i) => {
        if (m.action < outputSize) {
          targetData[i][m.action] = m.reward;
        }
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
    }
  }

  clearMemory() {
    this.memory = [];
  }

  addToMemory(state: number[], action: number, reward: number, nextState: number[] | null) {
    const expectedSize = (this.model.inputs[0].shape[1] as number);
    if (state.length !== expectedSize) return;
    
    this.memory.push({ state, action, reward, nextState });
    if (this.memory.length > this.maxMemory) this.memory.shift();
  }
}
