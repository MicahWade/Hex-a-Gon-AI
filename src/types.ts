export type Player = 1 | 2;

export interface Coord {
  q: number;
  r: number;
}

export interface Move {
  player: Player;
  coord: Coord;
  turn: number;
  moveInTurn: number;
  timestamp: number;
}

export type NotationType = 'axial' | 'ring';
export type LogPosition = 'left' | 'right';
export type Theme = 'dark' | 'light' | 'amoled';

export type BoardState = Map<string, Player>;

export function coordToString(coord: Coord): string {
  return `${coord.q},${coord.r}`;
}

export function stringToCoord(s: string): Coord {
  const [q, r] = s.split(',').map(Number);
  return { q, r };
}
