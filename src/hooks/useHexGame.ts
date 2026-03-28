import { useState, useCallback } from 'react';
import type { Player, BoardState, Move } from '../types';
import { coordToString } from '../types';
import { checkWin } from '../gameLogic';

interface GameState {
  board: BoardState;
  history: Move[];
  turn: number;
  currentPlayer: Player;
  movesLeft: number;
  winner: Player | null;
}

export function useHexGame() {
  const [gameState, setGameState] = useState<GameState>({
    board: new Map(),
    history: [],
    turn: 1,
    currentPlayer: 1,
    movesLeft: 1,
    winner: null
  });

  const makeMove = useCallback((q: number, r: number) => {
    setGameState(prev => {
      if (prev.winner) return prev;

      // Rule: First move of game always becomes (0,0)
      const isFirstMove = prev.board.size === 0;
      const targetQ = isFirstMove ? 0 : q;
      const targetR = isFirstMove ? 0 : r;
      const key = coordToString({ q: targetQ, r: targetR });

      if (prev.board.has(key)) return prev;

      // 1. Update Board
      const nextBoard = new Map(prev.board);
      nextBoard.set(key, prev.currentPlayer);

      // 2. Update History
      const moveInTurn = isFirstMove ? 1 : (prev.currentPlayer === 1 && prev.turn === 1 ? 1 : (3 - prev.movesLeft));
      const newMove: Move = {
        player: prev.currentPlayer,
        coord: { q: targetQ, r: targetR },
        turn: prev.turn,
        moveInTurn: moveInTurn,
        timestamp: Date.now()
      };
      const nextHistory = [...prev.history, newMove];

      // 3. Check Win
      if (checkWin(nextBoard, targetQ, targetR, prev.currentPlayer)) {
        return {
          ...prev,
          board: nextBoard,
          history: nextHistory,
          winner: prev.currentPlayer
        };
      }

      // 4. Update Turn Logic
      let nextPlayer = prev.currentPlayer;
      let nextTurn = prev.turn;
      let nextMovesLeft = prev.movesLeft - 1;

      if (nextMovesLeft === 0) {
        nextPlayer = prev.currentPlayer === 1 ? 2 : 1;
        nextTurn = prev.turn + (prev.currentPlayer === 2 ? 1 : (prev.turn === 1 ? 1 : 0));
        // Special case: Turn 1 is only 1 move for P1. All other turns are 2 moves.
        nextMovesLeft = 2;
      }

      return {
        board: nextBoard,
        history: nextHistory,
        turn: nextTurn,
        currentPlayer: nextPlayer,
        movesLeft: nextMovesLeft,
        winner: null
      };
    });
  }, []);

  const resetGame = useCallback(() => {
    setGameState({
      board: new Map(),
      history: [],
      turn: 1,
      currentPlayer: 1,
      movesLeft: 1,
      winner: null
    });
  }, []);

  return {
    board: gameState.board,
    history: gameState.history,
    turn: gameState.turn,
    currentPlayer: gameState.currentPlayer,
    movesLeftInTurn: gameState.movesLeft,
    winner: gameState.winner,
    makeMove,
    resetGame
  };
}
