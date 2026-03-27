import { useState, useCallback } from 'react';
import type { Player, BoardState } from '../types';
import { coordToString } from '../types';
import { checkWin } from '../gameLogic';

export function useHexGame() {
  const [board, setBoard] = useState<BoardState>(new Map());
  const [turn, setTurn] = useState<number>(1);
  const [currentPlayer, setCurrentPlayer] = useState<Player>(1);
  const [movesLeftInTurn, setMovesLeftInTurn] = useState<number>(1);
  const [winner, setWinner] = useState<Player | null>(null);

  const makeMove = useCallback((q: number, r: number) => {
    if (winner) return;
    
    const key = coordToString({ q, r });
    if (board.has(key)) return;

    const newBoard = new Map(board);
    newBoard.set(key, currentPlayer);
    setBoard(newBoard);

    if (checkWin(newBoard, q, r, currentPlayer)) {
      setWinner(currentPlayer);
      return;
    }

    if (movesLeftInTurn > 1) {
      setMovesLeftInTurn(movesLeftInTurn - 1);
    } else {
      // Switch player
      const nextPlayer = currentPlayer === 1 ? 2 : 1;
      setCurrentPlayer(nextPlayer);
      setTurn(prevTurn => prevTurn + 1);
      // Turn 1: 1 move for P1. All subsequent turns have 2 moves.
      setMovesLeftInTurn(2);
    }
  }, [board, currentPlayer, movesLeftInTurn, winner]);

  const resetGame = useCallback(() => {
    setBoard(new Map());
    setTurn(1);
    setCurrentPlayer(1);
    setMovesLeftInTurn(1);
    setWinner(null);
  }, []);

  return {
    board,
    turn,
    currentPlayer,
    movesLeftInTurn,
    winner,
    makeMove,
    resetGame
  };
}
