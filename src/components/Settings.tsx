import React from 'react';
import type { NotationType, LogPosition, Theme } from '../types';

interface Props {
  notation: NotationType;
  setNotation: (type: NotationType) => void;
  logPosition: LogPosition;
  setLogPosition: (pos: LogPosition) => void;
  theme: Theme;
  setTheme: (t: Theme) => void;
}

export const Settings: React.FC<Props> = ({ 
  notation, setNotation, 
  logPosition, setLogPosition,
  theme, setTheme 
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
      </div>
    </div>
  );
};
