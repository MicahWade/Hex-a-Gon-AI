# ⬢ Hex-A-Gon AI

Hex-A-Gon is a high-performance, reinforcement learning environment for an infinite hexagonal connection game. This project features a standardized **7x3500 Dueling DQN** architecture that can be trained either directly in your browser or via a high-speed Python factory.

## 🚀 Training Options

### Option 1: Zero-Install (Web Browser)
The easiest way to start. Uses **WebGPU** or **WebGL** to train directly on your screen.
1.  Open the website.
2.  Go to the **AI Training Lab** tab.
3.  Click **Initialize New Model** and then **Start Training**.
4.  *Works on: All Modern Browsers (Chrome/Edge/Firefox).*

### Option 2: High-Performance (Python Factory)
Bypass browser limits and max out your dedicated graphics card. Best for training the massive 7x3500 brain.

#### 1. General Setup
```bash
cd python_trainer
python -m venv .venv
# Activate:
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
```

#### 2. Install Drivers for Your Hardware
Choose the command that matches your GPU:

| GPU Brand | OS | Install Command |
| :--- | :--- | :--- |
| **AMD** | **Linux (Fedora/Ubuntu)** | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.0` |
| **AMD** | **Windows** | `pip install torch-directml` |
| **Nvidia** | **Win/Linux** | `pip install torch` |
| **Apple M1/M2/M3** | **Mac** | `pip install torch` |

#### 3. Special Linux Fix (For AMD Users on Fedora/RHEL)
If you get a `libamdhip64.so` error on Linux, run this to clear the security flags:
```bash
sudo dnf install patchelf
find .venv -name "*.so*" -exec patchelf --clear-execstack {} +
```

#### 4. Run the Factory
```bash
pip install -r requirements.txt
python train.py
```

## 🔄 Syncing Python Models to the Web

To play against your GPU-trained brain in the browser:

1.  **Export from Python**:
    ```bash
    pip install onnx onnx2tf tensorflow-cpu tensorflowjs
    python export_to_web.py
    ```
2.  **Sync to Browser**:
    Open the website, go to the **AI Training** tab, and click **Sync Python Model**.
3.  **Play**:
    Load the `python-imported-brain` in the Model Vault.

## 🧠 Neural Architecture: 7x3500
This project uses a standardized strategic brain:
- **7 Hidden Layers**
- **3500 Neurons per Layer**
- **1372 Input Nodes** (Multi-Focal Vision)
- **1356 Output Nodes** (Strategic Action Space)

## 📜 License
GNU General Public License v3.0 (GPL-3.0)
