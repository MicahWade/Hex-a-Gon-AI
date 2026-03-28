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
    if (winner) return;
    
    // First move always becomes (0,0) regardless of where clicked
    const targetQ = board.size === 0 ? 0 : q;
    const targetR = board.size === 0 ? 0 : r;

    const key = coordToString({ q: targetQ, r: targetR });
    if (board.has(key)) return;

    // Enforce that only current player's color is placed
    const newMove: Move = {
      player: currentPlayer,
      coord: { q: targetQ, r: targetR },
      turn,
      moveInTurn: movesLeftInTurn === 1 ? (turn === 1 ? 1 : 2) : 1,
      timestamp: Date.now()
    };

    const newBoard = new Map(board);
    newBoard.set(key, currentPlayer);
    setBoard(newBoard);
    setHistory(prev => [...prev, newMove]);

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
  }, [board, currentPlayer, movesLeftInTurn, winner, turn]);

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
