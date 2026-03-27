import type { Coord, Player, BoardState, NotationType } from './types';
import { coordToString } from './types';

// Axial coordinates system (flat topped hexagons)
// q: column, r: row
// x = size * (3/2 * q)
// y = size * (sqrt(3)/2 * q + sqrt(3) * r)

export const SQRT3 = Math.sqrt(3);

export function getNotation(coord: Coord, type: NotationType): string {
  const { q, r } = coord;
  if (q === 0 && r === 0) return 'CENTER';

  if (type === 'ring') {
    // Ring number (radius)
    const ring = Math.max(Math.abs(q), Math.abs(r), Math.abs(-q - r));
    const letter = String.fromCharCode(64 + ring); // A=1, B=2...

    // Direction 0 is Right: (ring, 0)
    // Find index by following the ring boundary
    // Directions for flat-topped ring traversal:
    // (0,-1) -> (-1,0) -> (-1,1) -> (0,1) -> (1,0) -> (1,-1)
    const ringDirs = [
      { q: -1, r: 0 }, { q: 0, r: 1 }, { q: 1, r: 0 }, 
      { q: 1, r: -1 }, { q: 0, r: -1 }, { q: -1, r: 1 }
    ];
    
    // Start at (ring, 0) which is index 0
    let currQ = ring;
    let currR = 0;
    let index = 0;

    // Follow the ring clockwise to find index
    const steps = [
      { q: 0, r: -1 }, { q: -1, r: 0 }, { q: -1, r: 1 }, 
      { q: 0, r: 1 }, { q: 1, r: 0 }, { q: 1, r: -1 }
    ];

    for (let side = 0; side < 6; side++) {
      for (let step = 0; step < ring; step++) {
        if (currQ === q && currR === r) return `${letter}${index}`;
        currQ += steps[side].q;
        currR += steps[side].r;
        index++;
      }
    }
    return `${letter}?`; // Should not happen
  }

  return `(${q}, ${r})`;
}

export function hexToPixel(q: number, r: number, size: number): { x: number; y: number } {
  const x = size * (1.5 * q);
  const y = size * (SQRT3 / 2 * q + SQRT3 * r);
  return { x, y };
}

export function pixelToHex(x: number, y: number, size: number): Coord {
  const q = (2/3 * x) / size;
  const r = (-1/3 * x + SQRT3/3 * y) / size;
  return cubeRound(q, r, -q - r);
}

function cubeRound(fracQ: number, fracR: number, fracS: number): Coord {
  let q = Math.round(fracQ);
  let r = Math.round(fracR);
  let s = Math.round(fracS);

  const qDiff = Math.abs(q - fracQ);
  const rDiff = Math.abs(r - fracR);
  const sDiff = Math.abs(s - fracS);

  if (qDiff > rDiff && qDiff > sDiff) {
    q = -r - s;
  } else if (rDiff > sDiff) {
    r = -q - s;
  } else {
    s = -q - r;
  }
  return { q, r };
}

// 3 axes: (q, r), (q, s), (r, s)
// Directions for searching in axial coordinates
export const DIRECTIONS: Coord[] = [
  { q: 1, r: 0 }, { q: 1, r: -1 }, { q: 0, r: -1 },
  { q: -1, r: 0 }, { q: -1, r: 1 }, { q: 0, r: 1 }
];

// Check if there are 6 in a row of the same player starting from (q, r)
export function checkWin(board: BoardState, q: number, r: number, player: Player): boolean {
  // We only need to check in 3 directions (positive) because the negative directions are covered by searching backwards
  const searchDirs = [
    { q: 1, r: 0 },  // r-axis (q changes)
    { q: 0, r: 1 },  // q-axis (r changes)
    { q: 1, r: -1 }  // s-axis (q+r changes)
  ];

  for (const dir of searchDirs) {
    let count = 1;

    // Search forward
    for (let i = 1; i < 6; i++) {
      if (board.get(coordToString({ q: q + dir.q * i, r: r + dir.r * i })) === player) {
        count++;
      } else {
        break;
      }
    }

    // Search backward
    for (let i = 1; i < 6; i++) {
      if (board.get(coordToString({ q: q - dir.q * i, r: r - dir.r * i })) === player) {
        count++;
      } else {
        break;
      }
    }

    if (count >= 6) return true;
  }

  return false;
}
