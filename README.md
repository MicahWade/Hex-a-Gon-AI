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

## 🛠️ Tech Stack

- **Frontend**: React 19 (Vite)
- **Intelligence**: TensorFlow.js (Deep Q-Learning)
- **Persistence**: IndexedDB for local model storage

## 📥 Local Setup & Installation

Follow these steps to run the environment on your own machine:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/MicahWade/Hex-a-Gon-AI.git
    cd Hex-a-Gon-AI/Hex-A-Gon
    ```

2.  **Install Dependencies**:
    Make sure you have [Node.js](https://nodejs.org/) installed.
    ```bash
    npm install
    ```

3.  **Start the Development Server**:
    ```bash
    npm run dev
    ```
    The application will be available at `http://localhost:5173`.

4.  **Build for Production**:
    ```bash
    npm run build
    ```

## 🎮 How to Train Your First AI

1.  Navigate to the **Architecture** tab to set your brain size (Default: 1536x1536x2048).
2.  Go to the **AI Training Lab** tab.
3.  Type a name for your model and click **Initialize New Model**.
4.  Adjust the **Parallel Games** setting based on your hardware (16-32 is recommended for modern GPUs).
5.  Click **Start Training** and watch the "Gen" count rise!
6.  Once you're happy with the progress, click **Save** and head to the **Play Game** tab to challenge your creation in **PvAI** mode.

## 📜 License

GNU General Public License v3.0 (GPL-3.0)
