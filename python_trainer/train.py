import tensorflow as tf
import numpy as np
import os

# ==========================================
# 🛑 AMD GPU SETUP & WARNING 🛑
# ==========================================
# Standard `tensorflow` defaults to Nvidia CUDA.
# To ensure this maxes out your AMD GPU, you must install the specific AMD version:
# 
# 👉 For Windows: pip install tensorflow-directml-plugin
# 👉 For Linux: pip install tensorflow-rocm
# 👉 For Mac (M1/M2/M3): pip install tensorflow-metal
# ==========================================

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("WARNING: No GPU detected! Training on CPU will be very slow.")

INPUT_NODES = 1372  # Matches Multi-Focal Vision setup
OUTPUT_NODES = 1356 # Matches move choices

# HYPERPARAMETERS
BATCH_SIZE = 256    # High batch size maxes out GPU utilization

def build_massive_model():
    """
    Builds the high-performance 7 layers of 3500 neurons.
    """
    print("\n🧠 Building Massive 7x3500 Dueling DQN Model...")
    inputs = tf.keras.Input(shape=(INPUT_NODES,))
    x = inputs
    
    # The 7 Dense Layers of 3500
    for i in range(7):
        x = tf.keras.layers.Dense(3500, activation='relu', name=f'dense_{i}')(x)
        
    # Dueling Network separation
    value = tf.keras.layers.Dense(1, activation='linear', name='value')(x)
    advantage = tf.keras.layers.Dense(OUTPUT_NODES, activation='linear', name='advantage')(x)
    
    # Q = V + (A - mean(A))
    adv_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
    q_values = value + (advantage - adv_mean)
    
    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

def train_loop():
    model = build_massive_model()
    model.summary()
    print("\n🚀 Pushing data to GPU...")

    # For pure GPU stress-testing/benchmarking
    replay_buffer_states = np.random.rand(10000, INPUT_NODES).astype(np.float32)
    replay_buffer_targets = np.random.rand(10000, OUTPUT_NODES).astype(np.float32)

    # tf.data.Dataset forces the GPU to stay at 100% utilization
    dataset = tf.data.Dataset.from_tensor_slices((replay_buffer_states, replay_buffer_targets))
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    try:
        for epoch in range(1, 101):
            print(f"\n--- Generation {epoch * 1000} ---")
            model.fit(dataset, epochs=1, verbose=1)
            
            # Save weights periodically
            if epoch % 10 == 0:
                model.save('hex_model.keras')
                print("\n💾 Model saved to 'hex_model.keras'.")
                print("To use this in the browser game, run:")
                print("tensorflowjs_converter --input_format=keras hex_model.keras ../Hex-A-Gon/public/python_model")

    except KeyboardInterrupt:
        print("\nTraining stopped. Saving current progress...")
        model.save('hex_model.keras')

if __name__ == "__main__":
    train_loop()
