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
import { decodeMove } from './ai/encoder';
import { checkWin, getMaxLine } from './gameLogic';
import { coordToString } from './types';
import type { NotationType, LogPosition, Theme, Player, Coord } from './types';
import { getVaultMetadata } from './ai/modelVault';
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

  // Lifted Training States
  const [maxTurns, setMaxTurns] = useState(250);
  const [batchSize, setBatchSize] = useState(64);
  const [epsilon, setEpsilon] = useState(0.2);
  const [parallelGames, setParallelGames] = useState(4);

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

  // Auto-load last model on boot with Metadata restoration
  useEffect(() => {
    const autoLoad = async () => {
      try {
        // Try to find if there was a previously active model
        const vault = getVaultMetadata();
        // If we have a model name saved in localStorage from last session, use it
        // For now, let's look for 'hex-a-gon-model' as the default standard
        const lastModelName = localStorage.getItem('hexagon-last-model-name') || "hex-a-gon-model";
        
        const model = await tf.loadLayersModel(`indexeddb://${lastModelName}`);
        if (!model.optimizer) {
          model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['accuracy']
          });
        }
        sharedModelRef.current = model;
        sharedTrainerRef.current = new Trainer(model);
        setIsAiLoaded(true);
        setActiveModelName(lastModelName);

        // Restore Metadata if available (Backwards Compatible)
        const meta = vault.find(m => m.name === lastModelName);
        if (meta) {
          if (meta.hiddenLayers) setNetworkArchitecture(meta.hiddenLayers);
          if (meta.focalRadii) setFocalRadii(meta.focalRadii);
          if (meta.generation !== undefined) setGenerations(meta.generation);
          if (meta.maxTurns !== undefined) setMaxTurns(meta.maxTurns);
          if (meta.batchSize !== undefined) setBatchSize(meta.batchSize);
          if (meta.epsilon !== undefined) setEpsilon(meta.epsilon);
          // parallelGames isn't in meta yet, but we'll keep it as is or add it later
        }
      } catch (e) {
        console.log("No model to auto-load on boot.");
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
          const topMoves = await sharedTrainerRef.current!.getTopMoves(
            encodeState(board, currentPlayer, foci, focalRadii, turn, 100), 5
          );

          let bestMove: Coord | null = null;
          let bestMoveScore = -Infinity;

          for (const option of topMoves) {
            const candidate = decodeMove(option.idx, foci, focalRadii);
            const key = coordToString(candidate);
            if (board.has(key)) continue;

            const simBoard = new Map(board);
            simBoard.set(key, currentPlayer);

            if (checkWin(simBoard, candidate.q, candidate.r, currentPlayer)) {
              bestMove = candidate;
              break;
            }

            let score = option.prob;
            const otherPlayer = currentPlayer === 1 ? 2 : 1;
            const enemyMax = getMaxLine(board, candidate.q, candidate.r, otherPlayer);
            if (enemyMax >= 4) score += 1.0;

            if (score > bestMoveScore) {
              bestMoveScore = score;
              bestMove = candidate;
            }
          }

          if (bestMove) {
            makeMove(bestMove.q, bestMove.r);
            if (board.size > 0) {
              await new Promise(resolve => setTimeout(resolve, 400));
              const newBoard = new Map(board);
              newBoard.set(coordToString(bestMove), currentPlayer);
              const newFoci = [...foci];
              newFoci[0] = bestMove;
              
              const result2 = await sharedTrainerRef.current!.playTurn(
                newBoard, currentPlayer, newFoci, focalRadii, 
                { epsilon: 0 } as any, turn, 100
              );
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
              <div className={`active-ai-indicator ${!isAiLoaded ? 'no-model' : ''}`}>
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
          maxTurns={maxTurns}
          setMaxTurns={setMaxTurns}
          batchSize={batchSize}
          setBatchSize={setBatchSize}
          epsilon={epsilon}
          setEpsilon={setEpsilon}
          parallelGames={parallelGames}
          setParallelGames={setParallelGames}
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
