import { useState } from 'react';
import { useHexGame } from './hooks/useHexGame';
import { HexBoard } from './components/HexBoard';
import { Rules } from './components/Rules';
import { AITraining } from './components/AITraining';
import './App.css';

type Tab = 'play' | 'rules' | 'ai';
type GameMode = 'pvp' | 'pvai';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('play');
  const [gameMode, setGameMode] = useState<GameMode>('pvp');
  const {
    board,
    turn,
    currentPlayer,
    movesLeftInTurn,
    winner,
    makeMove,
    resetGame
  } = useHexGame();

  return (
    <div className="app-container">
      <nav className="main-nav">
        <button className={activeTab === 'play' ? 'active' : ''} onClick={() => setActiveTab('play')}>Play Game</button>
        <button className={activeTab === 'rules' ? 'active' : ''} onClick={() => setActiveTab('rules')}>Rules</button>
        <button className={activeTab === 'ai' ? 'active' : ''} onClick={() => setActiveTab('ai')}>AI Training</button>
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
          <HexBoard
            board={board}
            onMove={makeMove}
            currentPlayer={currentPlayer}
            winner={winner}
          />
        </div>
      )}

      {activeTab === 'rules' && <Rules />}
      {activeTab === 'ai' && <AITraining />}
    </div>
  );
}

export default App;
