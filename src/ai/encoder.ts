import { BoardState, Player, Coord, coordToString } from '../types';
import { getNotation } from '../gameLogic';

/**
 * Converts the board state into a flat array (Tensor input) based on focal windows.
 */
export function encodeState(
  board: BoardState,
  currentPlayer: Player,
  foci: Coord[], // [Global, Self, P1_L1, P1_L2, P2_L1, P2_L2]
  radii: { global: number; self: number; memory: number }
): number[] {
  const input: number[] = [];

  // 1. Spatial Encoding for each Focal Window
  foci.forEach((focus, i) => {
    let radius = radii.memory;
    if (i === 0) radius = radii.global;
    if (i === 1) radius = radii.self;

    // Traversal logic for hexagonal ring-index to ensure consistent mapping
    // We traverse from ring 0 outwards to maintain the 1-to-1 mapping
    
    // Ring 0
    const centerKey = coordToString(focus);
    input.push(encodeHex(board.get(centerKey), currentPlayer));

    // Rings 1 to N
    const steps = [
      { q: 0, r: -1 }, { q: -1, r: 0 }, { q: -1, r: 1 }, 
      { q: 0, r: 1 }, { q: 1, r: 0 }, { q: 1, r: -1 }
    ];

    for (let r = 1; r <= radius; r++) {
      let currQ = focus.q + r;
      let currR = focus.r;
      for (let side = 0; side < 6; side++) {
        for (let step = 0; step < r; step++) {
          const key = `${currQ},${currR}`;
          input.push(encodeHex(board.get(key), currentPlayer));
          currQ += steps[side].q;
          currR += steps[side].r;
        }
      }
    }
  });

  // 2. Localization Encoding (Axial Q,R for each focus)
  foci.forEach(focus => {
    // Normalized coordinates (assuming active play within a large but finite range for stability)
    input.push(focus.q / 100); 
    input.push(focus.r / 100);
  });

  // 3. Context Nodes
  input.push(currentPlayer === 1 ? 1 : 0); // Team ID
  input.push(1); // Constant 1
  input.push(0); // Constant 0

  return input;
}

function encodeHex(val: Player | undefined, currentPlayer: Player): number {
  if (val === undefined) return 0; // Empty
  return val === currentPlayer ? 1 : -1; // Friendly vs Enemy
}

/**
 * Decodes a single move from the output index back to a Coord.
 */
export function decodeMove(
  index: number,
  foci: Coord[],
  radii: { global: number; self: number; memory: number }
): Coord {
  // Logic to find which focal window the index belongs to and the relative coordinate
  // This mirrors the encodeState traversal
  let currentIndex = 0;
  
  for (let i = 0; i < foci.length; i++) {
    const focus = foci[i];
    let radius = radii.memory;
    if (i === 0) radius = radii.global;
    if (i === 1) radius = radii.self;

    const hexCount = 3 * radius * (radius + 1) + 1;
    
    if (index < currentIndex + hexCount) {
      // The move is in this window!
      const relativeIndex = index - currentIndex;
      return indexToCoord(focus, relativeIndex, radius);
    }
    currentIndex += hexCount;
  }

  return foci[0]; // Fallback
}

function indexToCoord(focus: Coord, index: number, radius: number): Coord {
  if (index === 0) return focus;
  
  let count = 1;
  const steps = [
    { q: 0, r: -1 }, { q: -1, r: 0 }, { q: -1, r: 1 }, 
    { q: 0, r: 1 }, { q: 1, r: 0 }, { q: 1, r: -1 }
  ];

  for (let r = 1; r <= radius; r++) {
    let currQ = focus.q + r;
    let currR = focus.r;
    for (let side = 0; side < 6; side++) {
      for (let step = 0; step < r; step++) {
        if (count === index) return { q: currQ, r: currR };
        currQ += steps[side].q;
        currR += steps[side].r;
        count++;
      }
    }
  }
  return focus;
}
