import { useState } from 'react';
import { useHexGame } from './hooks/useHexGame';
import { HexBoard } from './components/HexBoard';
import { Rules } from './components/Rules';
import { AITraining } from './components/AITraining';
import { NetworkViz } from './components/NetworkViz';
import { Settings } from './components/Settings';
import { MoveLog } from './components/MoveLog';
import type { NotationType, LogPosition, Theme } from './types';
import './App.css';

type Tab = 'play' | 'rules' | 'ai' | 'network' | 'history' | 'settings';
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
        <button className={activeTab === 'network' ? 'active' : ''} onClick={() => setActiveTab('network')}>Network</button>
        <button className={activeTab === 'ai' ? 'active' : ''} onClick={() => setActiveTab('ai')}>AI Training</button>
        <button className={activeTab === 'rules' ? 'active' : ''} onClick={() => setActiveTab('rules')}>Rules</button>
        <button className={activeTab === 'settings' ? 'active' : ''} onClick={() => setActiveTab('settings')}>Settings</button>
      </nav>

      {activeTab === 'play' && (
        <div className="game-view">
          <div className="ui-overlay">
            <h1>Hex-A-Gon</h1>
            <div className="game-mode-selector">
              <button 
                className={gameMode === 'pvp' ? 'active-mode' : ''} 
                onClick={() => setGameMode('pvp')}
              >
                PvP
              </button>
              <button 
                className={gameMode === 'pvai' ? 'active-mode' : ''} 
                onClick={() => setGameMode('pvai')}
              >
                PvAI
              </button>
            </div>
            <div className="status">
              {winner ? (
                <div className="winner-announcement">
                  <h2 className={winner === 1 ? 'p1' : 'p2'}>
                    Player {winner} Wins!
                  </h2>
                  <button onClick={resetGame}>New Game</button>
                </div>
              ) : (
                <div className="turn-info">
                  <p>
                    Turn {turn} - <span className={currentPlayer === 1 ? 'p1' : 'p2'}>
                      Player {currentPlayer}
                    </span>
                  </p>
                  <p>Moves left: {movesLeftInTurn}</p>
                </div>
              )}
            </div>
            <div className="instructions">
              <p>6 in a row to win.</p>
              <p>Click to place. Drag to pan. Scroll to zoom.</p>
            </div>
            {!winner && <button onClick={resetGame} className="reset-btn">Reset</button>}
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
            onMove={makeMove}
            currentPlayer={currentPlayer}
            winner={winner}
            p1Color={p1Color}
            p2Color={p2Color}
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
      {activeTab === 'ai' && <AITraining isTraining={isTraining} setIsTraining={setIsTraining} />}
      {activeTab === 'network' && <NetworkViz isTraining={isTraining} />}
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
