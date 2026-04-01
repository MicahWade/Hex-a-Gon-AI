import { coordToString } from './types';
import type { Coord, Player, BoardState } from './types';

export const SQRT3 = Math.sqrt(3);

export function checkWin(board: BoardState, q: number, r: number, player: Player): boolean {
  return getMaxLine(board, q, r, player) >= 6;
}

export function getMaxLine(board: BoardState, q: number, r: number, player: Player): number {
  const searchDirs = [
    { q: 1, r: 0 },
    { q: 0, r: 1 },
    { q: 1, r: -1 }
  ];

  let max = 0;
  for (const dir of searchDirs) {
    let count = 1;
    for (let i = 1; i < 6; i++) {
      if (board.get(coordToString({ q: q + dir.q * i, r: r + dir.r * i })) === player) count++;
      else break;
    }
    for (let i = 1; i < 6; i++) {
      if (board.get(coordToString({ q: q - dir.q * i, r: r - dir.r * i })) === player) count++;
      else break;
    }
    if (count > max) max = count;
  }
  return max;
}

export function getTacticalMove(board: BoardState, player: Player, blockChance: number = 0.5): Coord {
  const opponent = (player === 1 ? 2 : 1) as Player;
  const validMoves: Coord[] = [];
  
  // Scoping: Find all empty hexes adjacent to existing pieces or within a reasonable range (0,0)
  // For simplicity and speed, we'll scan a window around active pieces
  const coords = Array.from(board.keys()).map(k => {
    const [q, r] = k.split(',').map(Number);
    return { q, r };
  });

  const minQ = Math.min(...coords.map(c => c.q), 0) - 2;
  const maxQ = Math.max(...coords.map(c => c.q), 0) + 2;
  const minR = Math.min(...coords.map(c => c.r), 0) - 2;
  const maxR = Math.max(...coords.map(c => c.r), 0) + 2;

  const candidates: {coord: Coord, score: number}[] = [];

  for (let q = minQ; q <= maxQ; q++) {
    for (let r = minR; r <= maxR; r++) {
      const c = { q, r };
      if (!board.has(coordToString(c))) {
        let score = 0;
        const myMax = getMaxLine(board, q, r, player);
        const enemyMax = getMaxLine(board, q, r, opponent);

        // Priority 1: Win immediately
        if (myMax >= 6) score += 1000;
        
        // Priority 2: Block enemy win (if chance passes)
        if (enemyMax >= 5 && Math.random() < blockChance) score += 500;

        // Priority 3: Build lines
        if (myMax === 5) score += 100;
        if (myMax === 4) score += 50;
        if (myMax === 3) score += 10;

        // Priority 4: Block enemy growth
        if (enemyMax === 4 && Math.random() < blockChance) score += 40;

        // Add some noise
        score += Math.random() * 2;

        candidates.push({ coord: c, score });
      }
    }
  }

  if (candidates.length === 0) return { q: 0, r: 0 };
  return candidates.sort((a, b) => b.score - a.score)[0].coord;
}

export function rotateCoord(coord: Coord, times: number): Coord {
  let { q, r } = coord;
  for (let i = 0; i < times % 6; i++) {
    const newQ = -r;
    const newR = q + r;
    q = newQ;
    r = newR;
  }
  return { q, r };
}

export function rotateBoard(board: BoardState, times: number): BoardState {
  if (times % 6 === 0) return board;
  const newBoard = new Map<string, Player>();
  board.forEach((player, key) => {
    const [qStr, rStr] = key.split(',');
    const coord = { q: parseInt(qStr), r: parseInt(rStr) };
    const rotated = rotateCoord(coord, times);
    newBoard.set(coordToString(rotated), player);
  });
  return newBoard;
}
