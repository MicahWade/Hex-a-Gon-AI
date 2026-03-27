import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { BoardState, Player } from '../types';
import { stringToCoord } from '../types';
import { hexToPixel, pixelToHex, SQRT3 } from '../gameLogic';

interface Props {
  board: BoardState;
  onMove: (q: number, r: number) => void;
  currentPlayer: Player;
  winner: Player | null;
}

const BASE_HEX_SIZE = 30;

export const HexBoard: React.FC<Props> = ({ board, onMove, currentPlayer, winner }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewport, setViewport] = useState({ x: 0, y: 0, scale: 1 });
  const [hoverCoord, setHoverCoord] = useState<{ q: number, r: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

  const drawHex = (ctx: CanvasRenderingContext2D, q: number, r: number, size: number, player?: Player, isHovered?: boolean) => {
    const { x, y } = hexToPixel(q, r, size);
    
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i;
      const px = x + size * Math.cos(angle);
      const py = y + size * Math.sin(angle);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();

    if (player) {
      ctx.fillStyle = player === 1 ? '#3498db' : '#e74c3c';
      ctx.fill();
      ctx.strokeStyle = '#2c3e50';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else if (isHovered && !winner) {
      ctx.fillStyle = currentPlayer === 1 ? 'rgba(52, 152, 219, 0.3)' : 'rgba(231, 76, 60, 0.3)';
      ctx.fill();
      ctx.strokeStyle = '#95a5a6';
      ctx.lineWidth = 1;
      ctx.stroke();
    } else {
      ctx.strokeStyle = '#7f8c8d';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  };

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(canvas.width / 2 + viewport.x, canvas.height / 2 + viewport.y);

    const size = BASE_HEX_SIZE * viewport.scale;

    // Calculate visible range
    const halfW = canvas.width / 2;
    const halfH = canvas.height / 2;
    
    const minX = -halfW - viewport.x;
    const maxX = halfW - viewport.x;
    const minY = -halfH - viewport.y;
    const maxY = halfH - viewport.y;

    // Convert pixel bounds to hex bounds
    const q1 = (2/3 * minX) / size;
    const r1 = (-1/3 * minX + SQRT3/3 * minY) / size;
    const q2 = (2/3 * maxX) / size;
    const r2 = (-1/3 * maxX + SQRT3/3 * maxY) / size;
    const q3 = (2/3 * minX) / size;
    const r3 = (-1/3 * minX + SQRT3/3 * maxY) / size;
    const q4 = (2/3 * maxX) / size;
    const r4 = (-1/3 * maxX + SQRT3/3 * minY) / size;

    const minQ = Math.floor(Math.min(q1, q2, q3, q4)) - 1;
    const maxQ = Math.ceil(Math.max(q1, q2, q3, q4)) + 1;
    const minR = Math.floor(Math.min(r1, r2, r3, r4)) - 1;
    const maxR = Math.ceil(Math.max(r1, r2, r3, r4)) + 1;

    // Draw grid
    for (let q = minQ; q <= maxQ; q++) {
      for (let r = minR; r <= maxR; r++) {
        const key = `${q},${r}`;
        const player = board.get(key);
        const isHovered = hoverCoord?.q === q && hoverCoord?.r === r;
        
        if (!player) {
          drawHex(ctx, q, r, size, undefined, isHovered);
        }
      }
    }

    // Draw placed pieces (all of them, just in case they are outside the grid loop but somehow visible)
    board.forEach((player, key) => {
      const { q, r } = stringToCoord(key);
      drawHex(ctx, q, r, size, player);
    });

    ctx.restore();
  }, [board, viewport, hoverCoord]);

  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current) {
        canvasRef.current.width = window.innerWidth;
        canvasRef.current.height = window.innerHeight;
        render();
      }
    };
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, [render]);

  useEffect(() => {
    render();
  }, [render]);

  const [mouseDownPos, setMouseDownPos] = useState({ x: 0, y: 0 });

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    const pos = { x: e.clientX, y: e.clientY };
    setLastMousePos(pos);
    setMouseDownPos(pos);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (isDragging) {
      const dx = e.clientX - lastMousePos.x;
      const dy = e.clientY - lastMousePos.y;
      setViewport(v => ({ ...v, x: v.x + dx, y: v.y + dy }));
      setLastMousePos({ x: e.clientX, y: e.clientY });
    } else {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left - canvas.width / 2 - viewport.x;
      const py = e.clientY - rect.top - canvas.height / 2 - viewport.y;
      const size = BASE_HEX_SIZE * viewport.scale;
      const coord = pixelToHex(px, py, size);
      setHoverCoord(coord);
    }
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    setIsDragging(false);
    
    // Check if we moved significantly
    const dx = e.clientX - mouseDownPos.x;
    const dy = e.clientY - mouseDownPos.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance < 5 && !winner) {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left - canvas.width / 2 - viewport.x;
      const py = e.clientY - rect.top - canvas.height / 2 - viewport.y;
      const size = BASE_HEX_SIZE * viewport.scale;
      const { q, r } = pixelToHex(px, py, size);
      onMove(q, r);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    const scaleFactor = 1.1;
    const newScale = e.deltaY < 0 ? viewport.scale * scaleFactor : viewport.scale / scaleFactor;
    // Limit zoom
    const clampedScale = Math.max(0.2, Math.min(5, newScale));
    
    setViewport(v => ({ ...v, scale: clampedScale }));
  };

  return (
    <canvas
      ref={canvasRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onWheel={handleWheel}
      onContextMenu={(e) => e.preventDefault()}
      style={{ display: 'block', backgroundColor: '#2c3e50', cursor: isDragging ? 'grabbing' : 'crosshair' }}
    />
  );
};
