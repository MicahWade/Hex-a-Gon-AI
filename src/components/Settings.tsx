import React from 'react';
import type { NotationType, LogPosition, Theme } from '../types';

interface Props {
  notation: NotationType;
  setNotation: (type: NotationType) => void;
  logPosition: LogPosition;
  setLogPosition: (pos: LogPosition) => void;
  theme: Theme;
  setTheme: (t: Theme) => void;
  p1Color: string;
  setP1Color: (c: string) => void;
  p2Color: string;
  setP2Color: (c: string) => void;
}

export const Settings: React.FC<Props> = ({ 
  notation, setNotation, 
  logPosition, setLogPosition,
  theme, setTheme,
  p1Color, setP1Color,
  p2Color, setP2Color
}) => {
  return (
    <div className="tab-content settings-view">
      <div className="settings-header">
        <h2>Settings</h2>
      </div>
      
      <div className="settings-grid">
        <section className="settings-section card">
          <h3>Game Notation</h3>
          <p className="section-desc">Coordinate system for move history.</p>
          
          <div className="button-group-vertical">
            <button 
              className={notation === 'axial' ? 'active-btn' : 'inactive-btn'} 
              onClick={() => setNotation('axial')}
            >
              Axial (q, r)
            </button>
            <button 
              className={notation === 'ring' ? 'active-btn' : 'inactive-btn'} 
              onClick={() => setNotation('ring')}
            >
              Ring-Index (A0, B6...)
            </button>
          </div>
        </section>

        <section className="settings-section card">
          <h3>UI Layout</h3>
          <p className="section-desc">Position of the Recent Moves log.</p>
          
          <div className="button-group-vertical">
            <button 
              className={logPosition === 'left' ? 'active-btn' : 'inactive-btn'} 
              onClick={() => setLogPosition('left')}
            >
              Bottom Left
            </button>
            <button 
              className={logPosition === 'right' ? 'active-btn' : 'inactive-btn'} 
              onClick={() => setLogPosition('right')}
            >
              Bottom Right
            </button>
          </div>
        </section>

        <section className="settings-section card">
          <h3>Theme</h3>
          <p className="section-desc">Visual appearance of the application.</p>
          
          <div className="button-group-vertical">
            <button 
              className={theme === 'dark' ? 'active-btn' : 'inactive-btn'} 
              onClick={() => setTheme('dark')}
            >
              Deep Blue (Default)
            </button>
            <button 
              className={theme === 'amoled' ? 'active-btn' : 'inactive-btn'} 
              onClick={() => setTheme('amoled')}
            >
              AMOLED Black
            </button>
          </div>
        </section>

        <section className="settings-section card">
          <h3>Player Colors</h3>
          <p className="section-desc">Customize hexagon colors.</p>
          
          <div className="color-config">
            <div className="color-input-item">
              <label>Player 1</label>
              <div className="picker-row">
                <input 
                  type="color" 
                  value={p1Color} 
                  onChange={(e) => setP1Color(e.target.value)} 
                />
                <span className="hex-label">{p1Color.toUpperCase()}</span>
              </div>
            </div>
            <div className="color-input-item" style={{ marginTop: '15px' }}>
              <label>Player 2</label>
              <div className="picker-row">
                <input 
                  type="color" 
                  value={p2Color} 
                  onChange={(e) => setP2Color(e.target.value)} 
                />
                <span className="hex-label">{p2Color.toUpperCase()}</span>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
