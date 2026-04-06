import numpy as np
import torch
import os
import shutil
import sys

# 1. NumPy Legacy Patch (Required for the converter)
import builtins
if not hasattr(np, 'object'): np.object = builtins.object
if not hasattr(np, 'bool'): np.bool = builtins.bool

import tensorflow as tf
# Force use of tf_keras for conversion
try:
    import tf_keras as keras
except ImportError:
    import tensorflow.keras as keras

# Import architecture info from train.py
from train import DuelingDQN, INPUT_NODES, OUTPUT_NODES

def export():
    print("📦 Starting ULTRA-LIGHT Weight Transplant (PyTorch -> TFJS)...")
    
    # 1. Load PyTorch Weights
    checkpoint_path = 'hex_brain.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: {checkpoint_path} not found.")
        return

    print("📂 Reading PyTorch weights...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pt_state = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint

    # 2. Build Identical Keras Model
    print("🧠 Building empty Keras mirror...")
    inputs = keras.Input(shape=(INPUT_NODES,))
    x = inputs
    
    # Feature Extractor (7 Layers)
    for i in range(7):
        x = keras.layers.Dense(3500, activation='relu', name=f'dense_{i}')(x)
    
    # Dueling Heads
    value = keras.layers.Dense(1, activation='linear', name='value_head')(x)
    advantage = keras.layers.Dense(OUTPUT_NODES, activation='linear', name='adv_head')(x)
    
    # Simple Q-value combination (Doesn't need complex lambda for weight saving)
    q_values = keras.layers.Add()([value, advantage]) 
    
    keras_model = keras.Model(inputs=inputs, outputs=q_values)

    # 3. TRANSPLANT WEIGHTS
    print("💉 Transplanting weights (This uses very little RAM)...")
    
    # Map PyTorch layers to Keras layers
    # PyTorch linear weights are [out, in], Keras are [in, out]
    
    # Feature Layers
    for i in range(7):
        # PyTorch Sequential index: Linear is at 0, 2, 4, 6, 8, 10, 12
        pt_idx = i * 2
        w = pt_state[f'feature_extractor.{pt_idx}.weight'].numpy().T
        b = pt_state[f'feature_extractor.{pt_idx}.bias'].numpy()
        keras_model.get_layer(f'dense_{i}').set_weights([w, b])

    # Head Layers
    w_v = pt_state['value_stream.weight'].numpy().T
    b_v = pt_state['value_stream.bias'].numpy()
    keras_model.get_layer('value_head').set_weights([w_v, b_v])

    w_a = pt_state['advantage_stream.weight'].numpy().T
    b_a = pt_state['advantage_stream.bias'].numpy()
    keras_model.get_layer('adv_head').set_weights([w_a, b_a])

    # 4. Convert to TFJS
    print("🛠️  Converting to Web Format...")
    tfjs_output_dir = "../Hex-A-Gon/public/python_model"
    if os.path.exists(tfjs_output_dir): shutil.rmtree(tfjs_output_dir)

    # Save to temp H5 first (Very RAM efficient)
    temp_h5 = "temp_model.h5"
    keras_model.save(temp_h5)

    try:
        from tensorflowjs.converters.converter import pip_main as tfjs_converter
        sys.argv = [
            'tensorflowjs_converter',
            '--input_format=keras',
            '--weight_shard_size_bytes=4194304', # 4MB shards
            temp_h5,
            tfjs_output_dir
        ]
        tfjs_converter()
        print(f"\n✨ SUCCESS! Model saved to: {tfjs_output_dir}")
    except Exception as e:
        print(f"❌ TFJS Conversion failed: {e}")
    finally:
        if os.path.exists(temp_h5): os.remove(temp_h5)

if __name__ == "__main__":
    export()
