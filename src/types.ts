export type Player = 1 | 2;

export interface Coord {
  q: number;
  r: number;
}

export type BoardState = Map<string, Player>;

export function coordToString(coord: Coord): string {
  return `${coord.q},${coord.r}`;
}

export function stringToCoord(s: string): Coord {
  const [q, r] = s.split(',').map(Number);
  return { q, r };
}
