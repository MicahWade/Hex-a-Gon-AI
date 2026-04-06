import sys
import types
import numpy as np
import builtins

# 1. THE "ULTIMATE" LEGACY PATCH
try:
    if not hasattr(np, 'object'): np.object = builtins.object
    if not hasattr(np, 'bool'): np.bool = builtins.bool
    if not hasattr(np, 'float'): np.float = builtins.float
    mock_estimator = types.ModuleType('estimator')
    mock_estimator.Exporter = object
    sys.modules['tensorflow.estimator'] = mock_estimator
    sys.modules['tensorflow.compat.v1.estimator'] = mock_estimator
    import tensorflow as tf
    tf.estimator = mock_estimator
    if not hasattr(tf, 'compat'): tf.compat = types.ModuleType('compat')
    if not hasattr(tf.compat, 'v1'): tf.compat.v1 = types.ModuleType('v1')
    tf.compat.v1.estimator = mock_estimator
    sys.modules['tensorflow_hub'] = types.ModuleType('tensorflow_hub')
    try:
        import tf_keras
        sys.modules['keras'] = tf_keras
    except ImportError: pass
except Exception as e:
    print(f"⚠️ Patching warning: {e}")

import torch
import torch.nn as nn
import os
import shutil
import subprocess

# Import the model structure from train.py
from train import DuelingDQN, INPUT_NODES, OUTPUT_NODES

# Import the converter
try:
    from tensorflowjs.converters.converter import pip_main as tfjs_converter
except ImportError:
    print("❌ Error: tensorflowjs not installed.")
    sys.exit(1)

def export():
    print("📦 Starting MEMORY-SAFE PyTorch to Web Conversion...")
    
    # 1. Load Model (Force CPU to save GPU memory for conversion)
    model = DuelingDQN()
    checkpoint_path = 'hex_brain.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: {checkpoint_path} not found.")
        return

    print(f"📂 Loading weights into system RAM...")
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
    # Using a simpler export to save RAM
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=12, do_constant_folding=True)
    
    # CLEAR RAM: Delete PyTorch model immediately after ONNX is saved
    del model
    del checkpoint
    import gc
    gc.collect()

    # 3. Convert ONNX to SavedModel (Intermediate)
    saved_model_dir = "temp_saved_model"
    tfjs_output_dir = "../Hex-A-Gon/public/python_model"

    try:
        if os.path.exists(saved_model_dir): shutil.rmtree(saved_model_dir)
        os.makedirs(saved_model_dir, exist_ok=True)
        if os.path.exists(tfjs_output_dir): shutil.rmtree(tfjs_output_dir)
        
        print("  > Phase A: ONNX to SavedModel (RAM Intensive)...")
        # We use simpler flags to reduce memory overhead during translation
        subprocess.run([
            "onnx2tf", 
            "-i", onnx_path, 
            "-o", saved_model_dir, 
            "-tb", "tf_converter",
            "-osd",
            "-nlt",
            "--not_use_onnxsim", # Skip optimization to save RAM
            "-v", "error"
        ], check=True)

        # 4. Search for the output
        print("  🔍 Scanning for generated files...")
        target_pb = None
        for root, dirs, files in os.walk(saved_model_dir):
            if "saved_model.pb" in files:
                target_pb = root
                break
        
        if not target_pb:
            print(f"❌ Error: Conversion failed to produce files.")
            return

        # 5. Convert to TFJS (Internal API)
        print("  > Phase B: SavedModel to TFJS (Final Repack)...")
        # Use sharding to keep browser memory usage low
        sys.argv = [
            'tensorflowjs_converter', 
            '--input_format=tf_saved_model', 
            '--weight_shard_size_bytes=4194304', # 4MB shards
            target_pb, 
            tfjs_output_dir
        ]
        tfjs_converter()

        print(f"\n✨ ALL DONE! Your massive brain is ready. ✨")
        print(f"Model saved to: {tfjs_output_dir}")

    except Exception as e:
        print(f"\n❌ Conversion Error: {e}")
    finally:
        # Final Cleanup
        if os.path.exists(saved_model_dir): shutil.rmtree(saved_model_dir)
        if os.path.exists(onnx_path): os.remove(onnx_path)

if __name__ == "__main__":
    export()
