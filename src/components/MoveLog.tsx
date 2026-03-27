import React, { useEffect, useRef } from 'react';
import type { Move, NotationType, LogPosition } from '../types';
import { getNotation } from '../gameLogic';

interface Props {
  history: Move[];
  notation: NotationType;
  isSidePanel?: boolean;
  position?: LogPosition;
  p1Color?: string;
  p2Color?: string;
}

export const MoveLog: React.FC<Props> = ({ 
  history, notation, isSidePanel, position = 'right',
  p1Color = '#3498db', p2Color = '#e74c3c'
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history]);

  const className = isSidePanel 
    ? `move-log-chat ${position === 'left' ? 'pos-left' : 'pos-right'}` 
    : 'tab-content full-history';

  return (
    <div className={className}>
      <div className="log-header">
        <h3>{isSidePanel ? 'Recent Moves' : 'Full Move History'}</h3>
      </div>
      <div className="move-list" ref={scrollRef}>
        {history.length === 0 ? (
          <p className="no-moves">No moves yet.</p>
        ) : (
          history.map((move, i) => (
            <div key={`${move.timestamp}-${i}`} className="move-entry">
              <span className="move-num">{move.turn}.{move.moveInTurn}</span>
              <span 
                className={`move-player p${move.player}`}
                style={{ color: move.player === 1 ? p1Color : p2Color }}
              >
                P{move.player}
              </span>
              <span className="move-coord">{getNotation(move.coord, notation)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
