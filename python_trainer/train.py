import tensorflow as tf
import numpy as np
import random
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
    print("WARNING: No GPU detected! Training 10 layers of 5000 on CPU will take weeks.")

INPUT_NODES = 1372  # Matches Multi-Focal Vision setup
OUTPUT_NODES = 1356 # Matches move choices

# HYPERPARAMETERS
BATCH_SIZE = 256    # High batch size maxes out GPU utilization

def build_massive_model():
    """
    Builds the user-requested 10 layers of 5000 neurons.
    WARNING: This creates ~250 Million Parameters and requires ~1.5GB to 3GB of VRAM.
    """
    print("\n🧠 Building Massive 10x5000 Dueling DQN Model...")
    inputs = tf.keras.Input(shape=(INPUT_NODES,))
    x = inputs
    
    # The 10 Dense Layers of 5000
    for i in range(10):
        x = tf.keras.layers.Dense(5000, activation='relu', name=f'dense_{i}')(x)
        
    # Dueling Network separation
    value = tf.keras.layers.Dense(1, activation='linear', name='value')(x)
    advantage = tf.keras.layers.Dense(OUTPUT_NODES, activation='linear', name='advantage')(x)
    
    # Q = V + (A - mean(A))
    adv_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
    q_values = value + (advantage - adv_mean)
    
    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

def build_natural_model():
    """
    "Build it Naturally" - Neural Architecture Search (NAS)
    Instead of hardcoding 10x5000, this naturally searches for the "best amount".
    It uses KerasTuner (or random bounds here) to find the perfect balance between
    speed and intelligence without blowing up your GPU memory.
    """
    print("\n🌱 Naturally exploring architecture for optimal size...")
    # For a true NAS, you would use `import keras_tuner as kt` and let it train 100 variations.
    # Here is an example of an auto-scaling layout that is more efficient:
    num_layers = random.randint(3, 6)
    neurons = random.randint(1000, 3000)
    print(f"Selected: {num_layers} layers of {neurons} neurons.")
    
    inputs = tf.keras.Input(shape=(INPUT_NODES,))
    x = inputs
    for i in range(num_layers):
        x = tf.keras.layers.Dense(neurons, activation='relu', name=f'dense_{i}')(x)
        
    value = tf.keras.layers.Dense(1, activation='linear')(x)
    advantage = tf.keras.layers.Dense(OUTPUT_NODES, activation='linear')(x)
    adv_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
    q_values = value + (advantage - adv_mean)
    
    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

def train_loop():
    # Toggle this to True if you want the program to naturally find the best size
    USE_NATURAL_SEARCH = False
    
    if USE_NATURAL_SEARCH:
        model = build_natural_model()
    else:
        model = build_massive_model()

    model.summary()
    print("\n🚀 Pushing data to GPU...")

    # NOTE: To do real training, you would implement the Hexagon logic in Python using NumPy
    # and feed the actual game experiences here. For pure GPU stress-testing, we use dummy data.
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
                print("💾 Model saved to 'hex_model.keras'.")
                print("To use this in the browser game, run:")
                print("pip install tensorflowjs")
                print("tensorflowjs_converter --input_format=keras hex_model.keras ../Hex-A-Gon/public/model")

    except KeyboardInterrupt:
        print("\nTraining stopped. Saving current progress...")
        model.save('hex_model.keras')

if __name__ == "__main__":
    train_loop()
