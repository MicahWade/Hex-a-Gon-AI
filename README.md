# Hex-A-Gon

A strategic, infinite-grid hexagonal "6-in-a-row" game built with React, TypeScript, and HTML5 Canvas.

## 🎮 Game Rules

Hex-A-Gon is a variation of connection games played on an infinite hexagonal grid with unique turn-based mechanics:

- **Objective**: Be the first player to connect **six** hexagons in a straight line along any of the three axes.
- **Asymmetric Opening**: 
  - **Turn 1**: Player 1 places **one** piece.
  - **Turn 2+**: Every subsequent turn, the active player places **two** pieces consecutively.
- **Infinite Board**: There are no boundaries. The game expands in any direction as players place their pieces.

## 🚀 Features

- **Infinite Canvas**: Drag to pan and scroll to zoom. The board scales with your strategy.
- **Axial Coordinate System**: Precise hexagonal math ensuring perfect placement and win detection.
- **Responsive UI**: Real-time turn tracking and move counters.
- **Performance**: Optimized Canvas rendering with viewport culling.

## 🛠️ Technical Stack

- **Framework**: [React 19](https://react.dev/)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **Build Tool**: [Vite](https://vite.dev/)
- **Rendering**: HTML5 Canvas API

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone git@github.com:MicahWade/Hex-a-Gon-AI.git
   cd Hex-a-Gon-AI
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Build for production**:
   ```bash
   npm run build
   ```

## 🕹️ Controls

- **Left Click**: Place a piece on an empty hex.
- **Left Click + Drag**: Pan around the infinite board.
- **Mouse Wheel**: Zoom in and out.
- **Right Click**: Disabled (to prevent context menu interference).

---
Developed as a high-performance, strategic playground for hexagonal grid logic.
