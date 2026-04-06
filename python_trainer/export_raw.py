import torch
import numpy as np
import os
import shutil
import json

# Import architecture info from train.py
from train import DuelingDQN

def export_raw():
    print("🚀 Starting RAW WEIGHT EXPORT (Fast & Memory-Safe)...")
    
    # 1. Load PyTorch weights
    checkpoint_path = 'hex_brain.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: {checkpoint_path} not found.")
        return

    print("📂 Reading PyTorch checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pt_state = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint

    # 2. Setup Output Directory
    output_dir = "../Hex-A-Gon/public/python_model"
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("💉 Extracting layers to binary...")
    
    # We will save weights in a format the Browser can easily read
    manifest = []

    # Helper to save a tensor as binary
    def save_tensor(name, tensor):
        # TFJS expects [input, output] for dense weights, PyTorch is [output, input]
        if 'weight' in name:
            data = tensor.numpy().T.astype(np.float32)
        else:
            data = tensor.numpy().astype(np.float32)
            
        filename = f"{name.replace('.', '_')}.bin"
        data.tofile(os.path.join(output_dir, filename))
        manifest.append({
            "name": name,
            "file": filename,
            "shape": list(data.shape)
        })

    # Export Feature Layers
    for i in range(7):
        pt_idx = i * 2
        save_tensor(f"feature_{i}_w", pt_state[f'feature_extractor.{pt_idx}.weight'])
        save_tensor(f"feature_{i}_b", pt_state[f'feature_extractor.{pt_idx}.bias'])

    # Export Dueling Heads
    save_tensor("value_w", pt_state['value_stream.weight'])
    save_tensor("value_b", pt_state['value_stream.bias'])
    save_tensor("advantage_w", pt_state['advantage_stream.weight'])
    save_tensor("advantage_b", pt_state['advantage_stream.bias'])

    # 3. Save Manifest
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    print(f"\n✅ SUCCESS! Raw weights exported to {output_dir}")
    print("Action: Go to the website and click 'Sync Python Model'.")

if __name__ == "__main__":
    export_raw()
