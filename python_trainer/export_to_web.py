import sys
import types
import numpy as np
import builtins

# 1. THE "ULTIMATE" LEGACY PATCH
# We create a fake module structure to satisfy the old tensorflowjs code
try:
    # Fix NumPy
    if not hasattr(np, 'object'): np.object = builtins.object
    if not hasattr(np, 'bool'): np.bool = builtins.bool
    if not hasattr(np, 'float'): np.float = builtins.float

    # Create a dummy estimator structure
    mock_estimator = types.ModuleType('estimator')
    mock_estimator.Exporter = object # Provide a base class for Hub to inherit from
    
    # Inject it everywhere the converter might look
    sys.modules['tensorflow.estimator'] = mock_estimator
    sys.modules['tensorflow.compat.v1.estimator'] = mock_estimator
    
    import tensorflow as tf
    # Force bind to the TF module
    tf.estimator = mock_estimator
    if not hasattr(tf, 'compat'): tf.compat = types.ModuleType('compat')
    if not hasattr(tf.compat, 'v1'): tf.compat.v1 = types.ModuleType('v1')
    tf.compat.v1.estimator = mock_estimator

    # 2. HIDE TENSORFLOW_HUB
    # Hub is the one crashing. We'll replace it with a blank module 
    # because our simple model doesn't actually need it.
    sys.modules['tensorflow_hub'] = types.ModuleType('tensorflow_hub')

except Exception as e:
    print(f"⚠️ Patching warning: {e}")

import torch
import torch.nn as nn
import os
import shutil

# Import the model structure from train.py
from train import DuelingDQN, INPUT_NODES, OUTPUT_NODES

# Import the converter
try:
    from tensorflowjs.converters.converter import pip_main as tfjs_converter
except ImportError:
    print("❌ Error: tensorflowjs not installed.")
    sys.exit(1)

def export():
    print("📦 Starting PyTorch to Web Conversion...")
    
    # 1. Load Model
    model = DuelingDQN()
    checkpoint_path = 'hex_brain.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: {checkpoint_path} not found.")
        return

    print(f"📂 Loading weights...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Export to ONNX
    print("🚀 Exporting to ONNX...")
    onnx_path = "hex_brain.onnx"
    dummy_input = torch.randn(1, INPUT_NODES)
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=12)

    # 3. Convert ONNX to TensorFlow.js
    saved_model_dir = "temp_saved_model"
    tfjs_output_dir = "../Hex-A-Gon/public/python_model"

    try:
        if os.path.exists(saved_model_dir): shutil.rmtree(saved_model_dir)
        if os.path.exists(tfjs_output_dir): shutil.rmtree(tfjs_output_dir)
        
        # ONNX -> SavedModel
        print("  > Phase A: ONNX to SavedModel...")
        import subprocess
        subprocess.run(["onnx2tf", "-i", onnx_path, "-o", saved_model_dir, "--non_verbose"], check=True)

        # SavedModel -> TFJS
        print("  > Phase B: SavedModel to TFJS...")
        sys.argv = ['tensorflowjs_converter', '--input_format=tf_saved_model', saved_model_dir, tfjs_output_dir]
        tfjs_converter()

        print(f"\n✨ ALL DONE! Model saved to: {tfjs_output_dir}")
    except Exception as e:
        print(f"\n❌ Conversion Error: {e}")
    finally:
        if os.path.exists(saved_model_dir): shutil.rmtree(saved_model_dir)
        if os.path.exists(onnx_path): os.remove(onnx_path)

if __name__ == "__main__":
    export()
