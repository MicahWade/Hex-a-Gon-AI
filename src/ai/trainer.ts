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
  private maxMemory = 5000;

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
    const moveCount = board.size === 0 ? 1 : 2;
    for (let i = 0; i < moveCount; i++) {
      const state = encodeState(currentBoard, player, foci, radii, turn, maxTurns);
      const action = await this.predictAction(state, config.epsilon, modelToUse);
      const move = decodeMove(action, foci, radii);

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

  private async predictAction(state: number[], epsilon: number, modelOverride?: tf.LayersModel): Promise<number> {
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * state.length);
    }

    const model = modelOverride || this.model;
    return tf.tidy(() => {
      const input = tf.tensor2d([state]);
      const prediction = model.predict(input) as tf.Tensor;
      return prediction.argMax(1).dataSync()[0];
    });
  }

  /**
   * Trains the model on a random batch of experiences from memory.
   */
  async trainBatch(batchSize: number) {
    if (this.memory.length < batchSize) return null;

    // Filter memory to ensure only states matching current model input size are used
    const expectedSize = (this.model.inputs[0].shape[1] as number);
    const validMemory = this.memory.filter(m => m.state.length === expectedSize);
    
    if (validMemory.length < batchSize) return null;

    const batch = validMemory.sort(() => 0.5 - Math.random()).slice(0, batchSize);
    
    // Efficiently stack tensors instead of mapping arrays of arrays
    const states = tf.tidy(() => tf.stack(batch.map(m => tf.tensor1d(m.state))));
    
    const targets = tf.tidy(() => {
      const currentPredictions = this.model.predict(states) as tf.Tensor;
      const targetData = currentPredictions.arraySync() as number[][];

      batch.forEach((m, i) => {
        targetData[i][m.action] = m.reward;
      });

      return tf.tensor2d(targetData);
    });

    const result = await this.model.fit(states, targets, { 
      epochs: 1, 
      batchSize: batchSize,
      verbose: 0 
    });
    
    states.dispose();
    targets.dispose();
    
    return result.history.loss[0] as number;
  }

  clearMemory() {
    this.memory = [];
  }

  addToMemory(state: number[], action: number, reward: number, nextState: number[] | null) {
    const expectedSize = (this.model.inputs[0].shape[1] as number);
    if (state.length !== expectedSize) return; // Ignore mismatched data
    
    this.memory.push({ state, action, reward, nextState });
    if (this.memory.length > this.maxMemory) this.memory.shift();
  }
}
