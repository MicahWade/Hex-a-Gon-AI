import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { useHexGame } from './hooks/useHexGame';
import { HexBoard } from './components/HexBoard';
import { Rules } from './components/Rules';
import { AITraining } from './components/AITraining';
import { ModelConfig } from './components/ModelConfig';
import { Settings } from './components/Settings';
import { MoveLog } from './components/MoveLog';
import { Trainer } from './ai/trainer';
import { encodeState, decodeMove } from './ai/encoder';
import { checkWin, getMaxLine } from './gameLogic';
import { coordToString } from './types';
import type { NotationType, LogPosition, Theme, Player, Coord } from './types';
import './App.css';

type Tab = 'play' | 'rules' | 'ai' | 'architecture' | 'history' | 'settings';
type GameMode = 'pvp' | 'pvai';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('play');
  const [gameMode, setGameMode] = useState<GameMode>('pvp');
  const [notation, setNotation] = useState<NotationType>('axial');
  const [logPosition, setLogPosition] = useState<LogPosition>('right');
  const [theme, setTheme] = useState<Theme>('dark');
  const [p1Color, setP1Color] = useState('#3498db');
  const [p2Color, setP2Color] = useState('#e74c3c');
  const [isTraining, setIsTraining] = useState(false);
  const [networkArchitecture, setNetworkArchitecture] = useState<number[]>([1536, 1536, 2048]);
  const [focalRadii, setFocalRadii] = useState({ global: 14, self: 8, memory: 6 });
  const [generations, setGenerations] = useState(0);
  const [loss, setLoss] = useState(0);
  const [targetDepth, setTargetDepth] = useState(3);
  const [activeModelName, setActiveModelName] = useState<string>("default-model");
  const [isAiLoaded, setIsAiLoaded] = useState(false);

  // SHARED AI INSTANCE
  const sharedModelRef = useRef<tf.LayersModel | null>(null);
  const sharedTrainerRef = useRef<Trainer | null>(null);

  // PvAI States
  const [userPlayer, setUserPlayer] = useState<Player>(1);
  const [gameStarted, setGameStarted] = useState(false);
  const aiProcessing = useRef(false);

  const {
    board,
    history,
    turn,
    currentPlayer,
    movesLeftInTurn,
    winner,
    makeMove,
    resetGame
  } = useHexGame();

  // Auto-load last model on boot
  useEffect(() => {
    const autoLoad = async () => {
      try {
        // Try to load the standard model
        const model = await tf.loadLayersModel('indexeddb://hex-a-gon-model');
        if (!model.optimizer) {
          model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
          });
        }
        sharedModelRef.current = model;
        sharedTrainerRef.current = new Trainer(model);
        setIsAiLoaded(true);
        setActiveModelName("hex-a-gon-model");
      } catch (e) {
        console.log("No default model to auto-load.");
      }
    };
    autoLoad();
  }, []);

  // Handle AI turn
  useEffect(() => {
    const isAiTurn = gameMode === 'pvai' && gameStarted && !winner && currentPlayer !== userPlayer && sharedTrainerRef.current;
    
    if (isAiTurn && !aiProcessing.current && !isTraining) {
      aiProcessing.current = true;
      
      const timer = setTimeout(async () => {
        const config = { epsilon: 0, rewards: {} } as any;
        const p1History = [...history].filter(m => m.player === 1).reverse();
        const p2History = [...history].filter(m => m.player === 2).reverse();
        const lastMove = history.length > 0 ? history[history.length - 1].coord : { q: 0, r: 0 };
        const aiLastMove = (userPlayer === 1 ? p2History : p1History)[0]?.coord || { q: 0, r: 0 };

        const foci: Coord[] = [
          lastMove, aiLastMove,
          p1History[0]?.coord || { q: 0, r: 0 },
          p1History[1]?.coord || { q: 0, r: 0 },
          p2History[0]?.coord || { q: 0, r: 0 },
          p2History[1]?.coord || { q: 0, r: 0 }
        ];

        try {
          // 1-Step Look-Ahead (Shallow MCTS)
          const topMoves = await sharedTrainerRef.current!.getTopMoves(encodeState(board, currentPlayer, foci, focalRadii, turn, 100), 5);

          let bestMove: Coord | null = null;
          let bestMoveScore = -Infinity;

          for (const option of topMoves) {
            const candidate = decodeMove(option.idx, foci, focalRadii);
            const key = coordToString(candidate);
            if (board.has(key)) continue;

            // Simulate making this move
            const simBoard = new Map(board);
            simBoard.set(key, currentPlayer);

            // Check if it wins immediately
            if (checkWin(simBoard, candidate.q, candidate.r, currentPlayer)) {
              bestMove = candidate;
              break; // Instant win, take it
            }

            // Simulate opponent's response
            let score = option.prob; // Base score is the initial probability
            const otherPlayer = currentPlayer === 1 ? 2 : 1;

            // Look for opponent's immediate threats
            // A simple check: does opponent have a 5-in-a-row now?
            // Since we can't easily simulate all opponent moves, we penalize if they had a threat we didn't block
            // The model's policy should handle this, but explicit lookahead helps.
            const enemyMax = getMaxLine(board, candidate.q, candidate.r, otherPlayer);
            if (enemyMax >= 4) {
              score += 1.0; // High bonus for blocking a major threat
            }

            if (score > bestMoveScore) {
              bestMoveScore = score;
              bestMove = candidate;
            }
          }

          if (bestMove) {
            makeMove(bestMove.q, bestMove.r);

            // For the second move, we just rely on the base model without deep lookahead for speed
            if (board.size > 0) {
              await new Promise(resolve => setTimeout(resolve, 400));
              const newBoard = new Map(board);
              newBoard.set(coordToString(bestMove), currentPlayer);

              const newFoci = [...foci];
              newFoci[0] = bestMove;

              const result2 = await sharedTrainerRef.current!.playTurn(newBoard, currentPlayer, newFoci, focalRadii, config, turn, 100);
              if (result2.moves.length > 0) {
                makeMove(result2.moves[0].q, result2.moves[0].r);
              }
            }
          }
        } catch (err) {
          console.error("AI Play Error:", err);
        } finally {
          aiProcessing.current = false;
        }
        }, 1000);

      return () => clearTimeout(timer);
    }
  }, [gameMode, gameStarted, currentPlayer, userPlayer, board, winner, turn, focalRadii, history, makeMove, isTraining]);

  const handleStartGame = () => {
    let finalUserPlayer = userPlayer;
    if ((userPlayer as any) === 'random') {
      finalUserPlayer = Math.random() > 0.5 ? 1 : 2;
      setUserPlayer(finalUserPlayer);
    }
    setGameStarted(true);
  };

  const handleReset = () => {
    resetGame();
    setGameStarted(false);
  };

  return (
    <div 
      className={`app-container theme-${theme}`}
      style={{ 
        '--p1-color': p1Color, 
        '--p2-color': p2Color 
      } as React.CSSProperties}
    >
      <nav className="main-nav">
        <button className={activeTab === 'play' ? 'active' : ''} onClick={() => setActiveTab('play')}>Play Game</button>
        <button className={activeTab === 'history' ? 'active' : ''} onClick={() => setActiveTab('history')}>History</button>
        <button className={activeTab === 'architecture' ? 'active' : ''} onClick={() => setActiveTab('architecture')}>Architecture</button>
        <button className={activeTab === 'ai' ? 'active' : ''} onClick={() => setActiveTab('ai')}>AI Training</button>
        <button className={activeTab === 'rules' ? 'active' : ''} onClick={() => setActiveTab('rules')}>Rules</button>
        <button className={activeTab === 'settings' ? 'active' : ''} onClick={() => setActiveTab('settings')}>Settings</button>
      </nav>

      {activeTab === 'play' && (
        <div className="game-view">
          <div className="ui-overlay">
            <h1>Hex-A-Gon</h1>
            {gameMode === 'pvai' && (
              <div className="active-ai-indicator">
                🤖 AI: <strong>{activeModelName}</strong>
                {!isAiLoaded && <p style={{color: '#e74c3c', fontSize: '10px', marginTop: '5px'}}>No model loaded. Go to AI tab to init/load.</p>}
              </div>
            )}
            <div className="game-mode-selector">
              <button className={gameMode === 'pvp' ? 'active-mode' : ''} onClick={() => { setGameMode('pvp'); setGameStarted(false); }}>PvP</button>
              <button className={gameMode === 'pvai' ? 'active-mode' : ''} onClick={() => { setGameMode('pvai'); setGameStarted(false); }}>PvAI</button>
            </div>

            {gameMode === 'pvai' && !gameStarted && (
              <div className="pvai-setup card">
                <h3>Select Your Side</h3>
                <div className="player-picker">
                  <button className={userPlayer === 1 ? 'active-p1' : ''} onClick={() => setUserPlayer(1)}>Blue (P1)</button>
                  <button className={userPlayer === 2 ? 'active-p2' : ''} onClick={() => setUserPlayer(2)}>Red (P2)</button>
                  <button className={(userPlayer as any) === 'random' ? 'active-rand' : ''} onClick={() => setUserPlayer('random' as any)}>Random</button>
                </div>
                <button className="start-game-btn" onClick={handleStartGame} disabled={!isAiLoaded}>Start Game</button>
              </div>
            )}

            {(gameMode === 'pvp' || gameStarted) && (
              <div className="status">
                {winner ? (
                  <div className="winner-announcement">
                    <h2 className={winner === 1 ? 'p1' : 'p2'}>Player {winner} Wins!</h2>
                    <button onClick={handleReset}>New Game</button>
                  </div>
                ) : (
                  <div className="turn-info">
                    <p>Turn {turn} - <span className={currentPlayer === 1 ? 'p1' : 'p2'}>Player {currentPlayer}</span></p>
                    <p>Moves left: {movesLeftInTurn}</p>
                  </div>
                )}
              </div>
            )}

            <div className="instructions">
              <p>6 in a row to win.</p>
              <p>Click to place. Drag to pan. Scroll to zoom.</p>
            </div>
            {!winner && <button onClick={handleReset} className="reset-btn">Reset</button>}
          </div>
          
          <MoveLog 
            history={history} 
            notation={notation} 
            isSidePanel={true} 
            position={logPosition} 
            p1Color={p1Color}
            p2Color={p2Color}
          />
          
          <HexBoard
            board={board}
            onMove={(q, r) => {
              if (gameMode === 'pvp') {
                makeMove(q, r);
              } else if (gameStarted && currentPlayer === userPlayer) {
                makeMove(q, r);
              }
            }}
            currentPlayer={currentPlayer}
            winner={winner}
            p1Color={p1Color}
            p2Color={p2Color}
            showHover={gameMode === 'pvp' || (gameStarted && currentPlayer === userPlayer)}
          />
        </div>
      )}

      {activeTab === 'history' && (
        <MoveLog 
          history={history} 
          notation={notation} 
          p1Color={p1Color}
          p2Color={p2Color}
        />
      )}
      {activeTab === 'rules' && <Rules />}
      {activeTab === 'ai' && (
        <AITraining 
          isTraining={isTraining} 
          setIsTraining={setIsTraining} 
          layers={networkArchitecture} 
          setLayers={setNetworkArchitecture}
          focalRadii={focalRadii}
          setFocalRadii={setFocalRadii}
          generations={generations}
          setGenerations={setGenerations}
          loss={loss}
          setLoss={setLoss}
          currentModelName={activeModelName}
          setCurrentModelName={setActiveModelName}
          trainerRef={sharedTrainerRef}
          modelRef={sharedModelRef}
          setIsAiLoaded={setIsAiLoaded}
        />
      )}
      {activeTab === 'architecture' && (
        <ModelConfig 
          layers={networkArchitecture} 
          setLayers={setNetworkArchitecture}
          focalRadii={focalRadii}
          setFocalRadii={setFocalRadii}
          targetDepth={targetDepth}
          setTargetDepth={setTargetDepth}
        />
      )}
      {activeTab === 'settings' && (
        <Settings 
          notation={notation} 
          setNotation={setNotation}
          logPosition={logPosition}
          setLogPosition={setLogPosition}
          theme={theme}
          setTheme={setTheme}
          p1Color={p1Color}
          setP1Color={setP1Color}
          p2Color={p2Color}
          setP2Color={setP2Color}
        />
      )}
    </div>
  );
}

export default App;
