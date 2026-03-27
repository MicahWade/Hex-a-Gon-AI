import React from 'react';

export const Rules: React.FC = () => {
  return (
    <div className="tab-content rules-view">
      <h2>Game Rules</h2>
      <section>
        <h3>Objective</h3>
        <p>The first player to connect <strong>six</strong> hexagons in a straight line along any of the three axes wins.</p>
      </section>

      <section>
        <h3>Asymmetric Opening</h3>
        <ul>
          <li><strong>Turn 1:</strong> Player 1 places <strong>one</strong> piece.</li>
          <li><strong>All Subsequent Turns:</strong> The active player places <strong>two</strong> pieces consecutively.</li>
        </ul>
        <p>This rule (known as the "pie rule" variation) is designed to balance the first-player advantage in connection games.</p>
      </section>

      <section>
        <h3>Controls</h3>
        <ul>
          <li><strong>Click:</strong> Place a piece.</li>
          <li><strong>Drag:</strong> Pan the infinite board.</li>
          <li><strong>Scroll:</strong> Zoom in and out.</li>
        </ul>
      </section>
      
      <section>
        <h3>AI Strategy Tip</h3>
        <p>Since the board is infinite, the game is more about <strong>denial and development</strong> than capturing territory. Focus on blocking opponent paths before they reach 4 in a row.</p>
      </section>
    </div>
  );
};
