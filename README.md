# ⬢ Hex-A-Gon AI

Hex-A-Gon is a high-performance, reinforcement learning environment for an infinite hexagonal connection game. Built with **React**, **TypeScript**, and **TensorFlow.js**, it allows users to design, train, and compete against advanced neural network architectures directly in the browser.

## 🚀 Advanced ML Features

- **High-Intensity Parallel Training**: Simulates 16-32 games simultaneously to maximize GPU utilization.
- **6x Data Augmentation**: Utilizes hexagonal rotational symmetry to multiply learning efficiency by 600%.
- **Dueling Deep Q-Network (DQN)**: Advanced architecture that separates state-value estimation from action advantage.
- **Tactical Instinct Rewards**: Immediate feedback system for building lines (3/4/5-in-a-row) and blocking opponent threats.
- **Prioritized Experience Replay (PER)**: Focuses training on the most critical strategic lessons rather than random moves.
- **Shallow MCTS Look-Ahead**: AI "thinks" before it moves, simulating future outcomes to avoid obvious traps.
- **Dynamic Learning Rate**: Exponentially decaying optimizer for fast initial learning and high-precision late-game polishing.

## 🧠 Neural Input Architecture

The AI perceives the world through a specialized **Multi-Focal Vision** system, using a total of **16 Metadata Nodes** for spatial and situational awareness:

### 1. Localization (12 Nodes)
Because the hexagonal grid is infinite, the AI uses 6 "Focal Points" to track activity. Each point provides its **Axial Q & R** coordinates to the network.
- **Global & Self Foci**: Tracks the center of the board and the AI's most recent move.
- **Tactical Memory Foci**: 4 dedicated points that "pin" historical battlegrounds, preventing the AI from forgetting distant threats while it explores new areas.

### 2. Contextual Awareness (4 Nodes)
- **Team ID**: Tells the AI if it is playing as P1 or P2.
- **The Bias (1.0)**: Provides a base activation signal for the network layers.
- **The Ground (0.0)**: A reference signal for weight normalization.
- **The Game Clock**: A normalized turn counter (0.0 to 1.0) that allows the AI to evolve its strategy from early-game expansion to late-game precision.

## 🛠️ Tech Stack

- **Frontend**: React 19 (Vite)
- **Intelligence**: TensorFlow.js (Deep Q-Learning)
- **Acceleration**: WebGPU (Primary) / WebGL (Fallback)
- **Multithreading**: Dedicated Web Workers for off-thread board math

## 📥 Local Setup & Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/MicahWade/Hex-a-Gon-AI.git
    cd Hex-a-Gon-AI/Hex-A-Gon
    ```

2.  **Install Dependencies**:
    ```bash
    npm install
    ```

3.  **Start the Development Server**:
    ```bash
    npm run dev
    ```

## 🎮 How to Train Your First AI

1.  Navigate to the **Architecture** tab to set your brain size (Default: Massive 5-layer config).
2.  Go to the **AI Training Lab** tab.
3.  Type a name for your model and click **Initialize New Model**.
4.  Adjust the **Parallel Games** (16-32 recommended).
5.  Click **Start Training** and watch the "Exploration Pulse" optimize your learning strategy.

## 📜 License

GNU General Public License v3.0 (GPL-3.0)
