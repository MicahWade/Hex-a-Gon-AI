import { useState, useCallback } from 'react';
import type { Player, BoardState, Move } from '../types';
import { coordToString } from '../types';
import { checkWin } from '../gameLogic';

export function useHexGame() {
  const [board, setBoard] = useState<BoardState>(new Map());
  const [history, setHistory] = useState<Move[]>([]);
  const [turn, setTurn] = useState<number>(1);
  const [currentPlayer, setCurrentPlayer] = useState<Player>(1);
  const [movesLeftInTurn, setMovesLeftInTurn] = useState<number>(1);
  const [winner, setWinner] = useState<Player | null>(null);

  const makeMove = useCallback((q: number, r: number) => {
    // Use functional updates to ensure we always have the latest state
    setBoard(currentBoard => {
      if (winner) return currentBoard;

      // First move always becomes (0,0)
      const targetQ = currentBoard.size === 0 ? 0 : q;
      const targetR = currentBoard.size === 0 ? 0 : r;
      const key = coordToString({ q: targetQ, r: targetR });

      if (currentBoard.has(key)) return currentBoard;

      // Create new board
      const nextBoard = new Map(currentBoard);
      nextBoard.set(key, currentPlayer);

      // We need to update other states based on this success
      // Note: We are inside setBoard, so we use functional updates for others too
      setHistory(prev => {
        const moveInTurn = currentBoard.size === 0 ? 1 : (3 - movesLeftInTurn);
        const newMove: Move = {
          player: currentPlayer,
          coord: { q: targetQ, r: targetR },
          turn,
          moveInTurn: moveInTurn,
          timestamp: Date.now()
        };
        return [...prev, newMove];
      });

      if (checkWin(nextBoard, targetQ, targetR, currentPlayer)) {
        setWinner(currentPlayer);
      } else {
        if (movesLeftInTurn > 1) {
          setMovesLeftInTurn(prev => prev - 1);
        } else {
          setCurrentPlayer(curr => (curr === 1 ? 2 : 1));
          setTurn(t => t + 1);
          setMovesLeftInTurn(2);
        }
      }

      return nextBoard;
    });
  }, [currentPlayer, movesLeftInTurn, winner, turn]);

  const resetGame = useCallback(() => {
    setBoard(new Map());
    setHistory([]);
    setTurn(1);
    setCurrentPlayer(1);
    setMovesLeftInTurn(1);
    setWinner(null);
  }, []);

  return {
    board,
    history,
    turn,
    currentPlayer,
    movesLeftInTurn,
    winner,
    makeMove,
    resetGame
  };
}
