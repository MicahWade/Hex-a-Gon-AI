import { encodeState, coordToIndex, decodeMove } from './encoder';
import { rotateBoard, rotateCoord } from '../gameLogic';
import type { Coord } from '../types';

/**
 * This worker handles the CPU-intensive task of rotating the board 6 times
 * and encoding the states for training.
 */
self.onmessage = (e) => {
  const { requestId, experiences, focalRadii, maxTurns, rewards } = e.data;
  const processedData: { state: number[], action: number, reward: number, priority: number }[] = [];

  for (const res of experiences) {
    for (const exp of res.experiences) {
      const priority = Math.abs(res.total) + 0.1;

      for (let r = 0; r < 6; r++) {
        // Original and 5 rotations
        const rotBoard = rotateBoard(exp.boardBefore, r);
        const rotFoci = exp.foci.map((f: Coord) => rotateCoord(f, r));
        const rotMove = rotateCoord(exp.move, r);
        const rotAction = coordToIndex(rotMove, rotFoci, focalRadii);
        
        if (rotAction !== -1) {
          const rotState = encodeState(rotBoard, exp.player, rotFoci, focalRadii, exp.turn, maxTurns);
          processedData.push({ state: rotState, action: rotAction, reward: res.total, priority });
        }

        // Penalty for high-ranking illegal moves
        for (const actionIdx of exp.illegalActions) {
          const rotIllegalMove = rotateCoord(decodeMove(actionIdx, exp.foci, focalRadii), r);
          const rotIllegalAction = coordToIndex(rotIllegalMove, rotFoci, focalRadii);
          if (rotIllegalAction !== -1) {
            const rotState = encodeState(rotBoard, exp.player, rotFoci, focalRadii, exp.turn, maxTurns);
            processedData.push({ state: rotState, action: rotIllegalAction, reward: rewards.illegal, priority });
          }
        }
      }
    }
  }

  // Send back with the same ID
  self.postMessage({ requestId, data: processedData });
};
