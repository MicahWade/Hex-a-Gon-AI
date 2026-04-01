# ⬢ Hex-A-Gon AI

Hex-A-Gon is a high-performance, reinforcement learning environment for an infinite hexagonal connection game. Built with **React**, **TypeScript**, and **TensorFlow.js**, it allows users to design, train, and compete against advanced neural network architectures directly in the browser.

## 🚀 Advanced ML Features

- **High-Intensity Parallel Training**: Simulates multiple games simultaneously to maximize GPU utilization.
- **6x Data Augmentation**: Utilizes hexagonal rotational symmetry to multiply learning efficiency by 600%.
- **Dueling Deep Q-Network (DQN)**: Robust architecture that separates state-value estimation from action advantage.
- **Tactical Instinct Rewards**: Immediate feedback system for building lines (3/4/5-in-a-row) and blocking opponent threats.
- **Prioritized Experience Replay (PER)**: Focuses training on the most critical strategic lessons rather than random moves.
- **Shallow MCTS Look-Ahead**: AI "thinks" before it moves, simulating future outcomes to avoid obvious traps.
- **Dynamic Learning Rate**: Exponentially decaying optimizer for fast initial learning and high-precision late-game polishing.

## 🛠️ Tech Stack

- **Frontend**: React 19 (Vite)
- **Engine**: Custom Hexagonal Coordinate System (Axial)
- **Intelligence**: TensorFlow.js (Deep Q-Learning)
- **Persistence**: IndexedDB for model weights and training metadata

## 🎮 How to Play

1. **PvP Mode**: Play against a local friend on an infinite grid.
2. **PvAI Mode**: Challenge your trained models.
3. **AI Lab**: Initialize new models, configure architecture (Hidden Layers, Focal Radii), and watch the AI evolve in real-time.

## 🧠 Training Tips

- **Starting Fresh**: When changing rewards or architecture, it is best to initialize a new model.
- **Parallel Games**: Increase the "Parallel" count to train faster if your GPU/CPU can handle it.
- **Save Names**: You can now rename and save models directly in the UI without browser prompts.

## 🏗️ Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## 📜 License

MIT
