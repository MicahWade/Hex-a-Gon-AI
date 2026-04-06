import numpy as np
import builtins

# 1. NumPy Legacy Patch
if not hasattr(np, 'object'):
    np.object = builtins.object
if not hasattr(np, 'bool'):
    np.bool = builtins.bool
if not hasattr(np, 'float'):
    np.float = builtins.float

import tensorflow as tf
import sys

# 2. TensorFlow Estimator Patch (Fixes the hub/estimator error)
try:
    import tensorflow_estimator
    if not hasattr(tf.compat.v1, 'estimator'):
        tf.compat.v1.estimator = tensorflow_estimator.estimator
except ImportError:
    pass

import torch
import torch.nn as nn
import os
import shutil

# Import the model structure from train.py
from train import DuelingDQN, INPUT_NODES, OUTPUT_NODES

# Import the converter's main function directly
try:
    from tensorflowjs.converters.converter import pip_main as tfjs_converter
except ImportError:
    print("❌ Error: tensorflowjs not installed. Run: pip install tensorflowjs")
    sys.exit(1)

def export():
    print("📦 Starting PyTorch to Web Conversion...")
    
    # 1. Initialize and Load Model
    model = DuelingDQN()
    checkpoint_path = 'hex_brain.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: {checkpoint_path} not found. Train the model first!")
        return

    print(f"📂 Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded Checkpoint (Gen {checkpoint.get('gen', 'Unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded Raw Weights")
    
    model.eval()

    # 2. Export to ONNX
    print("🚀 Exporting to ONNX...")
    onnx_path = "hex_brain.onnx"
    dummy_input = torch.randn(1, INPUT_NODES)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Created {onnx_path}")

    # 3. Convert ONNX to TensorFlow.js
    saved_model_dir = "temp_saved_model"
    tfjs_output_dir = "../Hex-A-Gon/public/python_model"

    try:
        # Clean up old temp dirs
        if os.path.exists(saved_model_dir): shutil.rmtree(saved_model_dir)
        if os.path.exists(tfjs_output_dir): shutil.rmtree(tfjs_output_dir)
        
        # ONNX -> SavedModel
        print("  > Phase A: ONNX to SavedModel...")
        import subprocess
        subprocess.run([
            "onnx2tf",
            "-i", onnx_path,
            "-o", saved_model_dir,
            "--non_verbose"
        ], check=True)

        # SavedModel -> TFJS
        print("  > Phase B: SavedModel to TFJS...")
        
        sys.argv = [
            'tensorflowjs_converter',
            '--input_format=tf_saved_model',
            saved_model_dir,
            tfjs_output_dir
        ]
        
        tfjs_converter()

        print(f"\n✨ ALL DONE! ✨")
        print(f"Model saved to: {tfjs_output_dir}")
        print("Action: Go to the website and click 'Sync Python Model'.")

    except Exception as e:
        print(f"\n❌ Conversion Error: {e}")
    finally:
        # Cleanup
        if os.path.exists(saved_model_dir): shutil.rmtree(saved_model_dir)
        if os.path.exists(onnx_path): os.remove(onnx_path)

if __name__ == "__main__":
    export()
