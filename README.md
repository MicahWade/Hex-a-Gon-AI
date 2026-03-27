# Hex-A-Gon

A strategic, infinite-grid hexagonal "6-in-a-row" game built with React, TypeScript, and TensorFlow.js.

## 🎮 Game Rules

Hex-A-Gon is a variation of connection games played on an infinite hexagonal grid with unique turn-based mechanics:

- **Objective**: Be the first player to connect **six** hexagons in a straight line along any of the three axes.
- **Asymmetric Opening**: 
  - **Turn 1**: Player 1 places **one** piece (strictly at 0,0).
  - **Turn 2+**: Every subsequent turn, the active player places **two** pieces consecutively.
- **Infinite Board**: There are no boundaries. The game expands in any direction as players place their pieces.

## 🧠 AI Training Guide

You can train your own Deep Learning model to play Hex-A-Gon directly in your browser.

### How to Start Training:

1.  **Configure Architecture**: Go to the **Architecture** tab. Design your model's hidden layers (Recommended: use the "Recommended" button for a balanced start).
2.  **Set Incentives**: Go to the **AI Training** tab and adjust the **Reward System**. This tells the AI what outcomes to value (e.g., favoring Player 2 wins or aggressive threat detection).
3.  **Launch**: Press **"Start Training"**. The AI will begin playing against itself in the background.
4.  **Monitor**:
    -   **Training Log**: Watch real-time game results.
    -   **Loss Graph**: A decreasing loss value indicates the AI is learning to predict moves more accurately.
    -   **History**: Completed games will appear in the History tab if enabled.

### Training Mechanics:
- **Self-Play**: The AI generates its own data by playing against its current version.
- **Multi-Focal Vision**: The AI "sees" the board through 6 high-resolution focal windows centered on recent activity.
- **Deep Q-Learning**: Uses reinforcement learning to optimize 6.5M+ parameters for winning strategies.

## 🚀 Features

- **Infinite Canvas**: Drag to pan and scroll to zoom.
- **Customization**: Change player colors, UI themes (Deep Blue, AMOLED, Dark), and move notations (Axial, RingIndex).
- **Architecture Lab**: Real-time complexity estimation and layer control.

## 🛠️ Technical Stack

- **Framework**: [React 19](https://react.dev/)
- **AI Engine**: [TensorFlow.js](https://www.tensorflow.org/js)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **Rendering**: HTML5 Canvas API

## 🕹️ Controls

- **Left Click**: Place a piece.
- **Left Click + Drag**: Pan the board.
- **Mouse Wheel**: Zoom.

---
Developed as a high-performance, strategic playground for hexagonal AI research.
