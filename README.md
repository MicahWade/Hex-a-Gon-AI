# Hex-A-Gon

A strategic, infinite-grid hexagonal "6-in-a-row" game built with React, TypeScript, and TensorFlow.js.

## 🧠 AI Training Lab (Browser-Based)

Hex-A-Gon features a fully integrated **Reinforcement Learning** environment. You can design, train, and test Deep Learning models entirely within your web browser.

### 📍 Where Training Happens
Training occurs **locally on your machine** using your browser's hardware acceleration (GPU via WebGL). 
- **Privacy**: No game data or models are ever uploaded to a server.
- **Performance**: High-speed self-play happens in a background loop within the application.

### 🛠️ How to Train:

1.  **Architecture**: Go to the **Architecture** tab.
    - Set your **Global Focus** (default R14) and **Self Focus** (default R8).
    - Use **Auto-Fill** to generate an optimized "Funnel" neural network for your desired depth.
2.  **Incentives**: Go to the **AI Training** tab.
    - Tune the **Reward System** (e.g., Threat Detection vs. Efficiency).
3.  **Execute**: Press **"Start Training"**. 
    - The **Training Log** will show real-time game results.
    - The **Loss** metric tracks the AI's learning progress (lower is better).

### 🔬 Advanced Architecture
- **Multi-Focal Vision**: The AI "sees" through 6 simultaneous focal windows (Global, Self, and 4 tactical memory windows).
- **Sequential 2-Move Logic**: The AI makes its two moves one-by-one, re-observing the board after the first move for better tactical awareness.
- **Mirror Selection**: The output layer is a 1-to-1 mapping of all observed hexes, making it highly efficient for spatial learning.

## 🎮 Game Rules
- **Objective**: Connect **six** hexagons in a straight line.
- **Opening**: Player 1 starts at (0,0).
- **Turns**: Every turn after the first consists of **two moves**.
- **Board**: Infinite expansion in all directions.

## 🛠️ Technical Stack
- **Framework**: React 19
- **AI Engine**: TensorFlow.js (Deep Q-Learning)
- **Rendering**: HTML5 Canvas API

---
Developed for high-performance hexagonal strategy and AI research.
