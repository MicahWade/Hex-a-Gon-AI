import { useHexGame } from './hooks/useHexGame';
import { HexBoard } from './components/HexBoard';
import './App.css';

function App() {
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
      <div className="ui-overlay">
        <h1>Hex-A-Gon</h1>
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
  );
}

export default App;
