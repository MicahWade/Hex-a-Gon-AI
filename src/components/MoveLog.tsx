import React, { useEffect, useRef } from 'react';
import type { Move, NotationType, LogPosition } from '../types';
import { getNotation } from '../gameLogic';

interface Props {
  history: Move[];
  notation: NotationType;
  isSidePanel?: boolean;
  position?: LogPosition;
}

export const MoveLog: React.FC<Props> = ({ history, notation, isSidePanel, position = 'right' }) => {
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
              <span className={`move-player p${move.player}`}>P{move.player}</span>
              <span className="move-coord">{getNotation(move.coord, notation)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
